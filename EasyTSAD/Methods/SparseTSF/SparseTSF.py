from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import tqdm
from torch import optim
from torch.utils.data import DataLoader

from ...DataFactory import TSData
from ...DataFactory.TorchDataSet.ReconstructWindow import (
    UTSAllInOneDataset,
    UTSOneByOneDataset,
)
from ...Exptools import EarlyStoppingTorch
from .. import BaseMethod


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, period_len):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.period_len = period_len

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2,
            padding_mode="zeros",
            bias=False,
        )

        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)

    def forward(self, x):
        # breakpoint()
        batch_size = x.shape[0]
        x = x.unsqueeze(-1)
        # normalization and permute b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # 1D convolution aggregation
        x = (
            self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(
                -1, self.enc_in, self.seq_len
            )
            + x
        )

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # sparse forecasting
        y = self.linear(x)  # bc,w,m

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean

        return y.squeeze(-1)


class SparseTSF(BaseMethod):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.__anomaly_score = None

        self.cuda = True
        if self.cuda == True and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("=== Using CUDA ===")
        else:
            if self.cuda == True and not torch.cuda.is_available():
                print("=== CUDA is unavailable ===")
            self.device = torch.device("cpu")
            print("=== Using CPU ===")

        self.seq_len = params["seq_len"]
        self.pred_len = params["pred_len"]
        self.enc_in = params["enc_in"]
        self.period_len = params["period_len"]

        self.step_size = self.pred_len // self.seq_len
        assert self.step_size * self.seq_len == self.pred_len

        self.batch_size = params["batch_size"]
        self.model = Model(
            self.seq_len, self.pred_len, self.enc_in, self.period_len
        ).to(self.device)
        self.epochs = params["epochs"]
        learning_rate = params["lr"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.75
        )
        self.loss = nn.MSELoss()

        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)

    def train_valid_phase(self, tsTrain: TSData):
        train_loader = DataLoader(
            dataset=UTSOneByOneDataset(tsTrain, "train", window_size=self.pred_len),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        valid_loader = DataLoader(
            dataset=UTSOneByOneDataset(tsTrain, "valid", window_size=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (x, target) in loop:
                x = x[:, :: self.step_size]
                x, target = x.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()

                output = self.model(x)
                loss = self.loss(output, target)
                loss.backward()

                self.optimizer.step()

                avg_loss += loss.cpu().item()
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            self.model.eval()
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(valid_loader), total=len(valid_loader), leave=True
            )
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x = x[:, :: self.step_size]
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f"Validation Epoch [{epoch}/{self.epochs}]")
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            valid_loss = avg_loss / max(len(valid_loader), 1)
            self.scheduler.step()

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break

    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        train_loader = DataLoader(
            dataset=UTSAllInOneDataset(tsTrains, "train", window_size=self.pred_len),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        valid_loader = DataLoader(
            dataset=UTSAllInOneDataset(tsTrains, "valid", window_size=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (x, target) in loop:
                x = x[:, :: self.step_size]
                x, target = x.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()

                output = self.model(x)
                loss = self.loss(output, target)
                loss.backward()

                self.optimizer.step()

                avg_loss += loss.cpu().item()
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            self.model.eval()
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(valid_loader), total=len(valid_loader), leave=True
            )
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x = x[:, :: self.step_size]
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f"Validation Epoch [{epoch}/{self.epochs}]")
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            valid_loss = avg_loss / max(len(valid_loader), 1)
            self.scheduler.step()

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break

    def test_phase(self, tsData: TSData):
        test_loader = DataLoader(
            dataset=UTSOneByOneDataset(tsData, "test", window_size=self.pred_len),
            batch_size=1,
            shuffle=False,
        )

        self.model.eval()
        scores = []
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:
                x = x[:, :: self.step_size]
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                # loss = self.loss(output, target)
                mse = torch.sub(output, target).pow(2)
                scores.append(mse.cpu()[:, -1])
                loop.set_description(f"Testing: ")

        scores = torch.cat(scores, dim=0)
        scores = scores.numpy().flatten()

        assert scores.ndim == 1
        self.__anomaly_score = scores

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(
            self.model, (self.batch_size, self.seq_len), verbose=0
        )
        with open(save_file, "w") as f:
            f.write(str(model_stats))
