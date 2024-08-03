from typing import Dict

import numpy as np
import torch as th
import torchinfo
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from EasyTSAD.DataFactory import TSData
from EasyTSAD.Exptools import EarlyStoppingTorch
from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory.TorchDataSet import PredictWindow
from EasyTSAD.DataFactory.TorchDataSet.PredictWindow import UTSOneByOneDataset
# from .TSDataset import UTSOneByOneDataset


# make it interpretable
class FANModel(nn.Module):
    def __init__(
        self, window: int, order: int, stacks: int | None = None, *args, **kwargs
    ) -> None:
        super().__init__()
        self.order = order
        self.channels = 2 * self.order + 1
        self.register_buffer(
            "orders",
            self._create_custom_periodic_cosine(window, self.order).unsqueeze(
                0
            ),  # (1, order, window)
        )
        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        # self.fc1 = nn.Linear(window, 1)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Conv1d(1, 1, window, padding=0, stride=1, dilation=1)

    def forward(
        self, x: th.Tensor, return_last: bool = False, *args, **kwargs
    ) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        # x = th.clip(x, 0, 1)
        res = []
        res.append(x.unsqueeze(1))
        ff = th.concat(
            [self.orders.repeat(x.size(0), 1, 1)]  # type: ignore
            + [th.cos(order * x.unsqueeze(1)) for order in range(1, self.order + 1)]
            + [x.unsqueeze(1)],
            dim=1,
        )  # batch,self.channel,window
        res.append(ff)
        ff = self.init_conv(ff)
        ff = self.bn1(ff)
        ff = self.act(ff)
        ff = self.inner_conv(ff) + res.pop()
        ff = self.bn2(ff)
        ff = self.act(ff)
        ff = self.out_conv(ff) + res.pop()
        ff = self.bn3(ff)
        ff = self.act(ff)
        ff = self.final_conv(ff)
        if return_last:
            return ff.squeeze(1), ff
        return ff.squeeze(1)
        # return (self.fc1(ff), ff) if return_last else (self.fc1(ff), None)

    def _create_custom_periodic_cosine(
        self, window: int, period: list[int] | int
    ) -> th.Tensor:
        d = len(period) if isinstance(period, list) else period
        pl = period if isinstance(period, list) else [i for i in range(1, period + 1)]
        # 初始化结果tensor
        result = th.empty(d, window, dtype=th.float32)
        # 为每个周期生成对应的cosine序列
        for i, p in enumerate(pl):
            t = th.arange(window, dtype=th.float32) / p * 2 * th.pi
            result[i, :] = th.cos(t)
        return result


class FAN(BaseMethod):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.__anomaly_score = None

        self.cuda = True
        if self.cuda is True and th.cuda.is_available():
            self.device = th.device("cuda")
            print("=== Using CUDA ===")
        else:
            if self.cuda is True and not th.cuda.is_available():
                print("=== CUDA is unavailable ===")
            self.device = th.device("cpu")
            print("=== Using CPU ===")

        self.batch_size = params["batch_size"]
        self.window = params["window"]
        self.debug = params.get("debug", False)
        self.model = FANModel(**params).to(self.device)

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
            dataset=UTSOneByOneDataset(tsTrain, "train", window_size=self.window),
            batch_size=self.batch_size,
            shuffle=True,
        )

        valid_loader = DataLoader(
            dataset=UTSOneByOneDataset(tsTrain, "valid", window_size=self.window),
            batch_size=self.batch_size,
            shuffle=False,
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (x, target) in loop:
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
            with th.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)
                    # breakpoint()
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
            dataset=PredictWindow.UTSAllInOneDataset(
                tsTrains, "train", window_size=self.window
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        valid_loader = DataLoader(
            dataset=PredictWindow.UTSAllInOneDataset(
                tsTrains, "valid", window_size=self.window
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (x, target) in loop:
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
            with th.no_grad():
                for idx, (x, target) in loop:
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
            dataset=UTSOneByOneDataset(tsData, "test", window_size=self.window),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model.eval()
        scores = []
        # lst_layer = []
        all_x = []
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with th.no_grad():
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x, return_last=self.debug)
                # loss = self.loss(output, target)
                mse = th.sub(output, target).abs()
                scores.append(mse.cpu())
                # lst_layer.append(last_layer.cpu())
                all_x.append(x.cpu())
                loop.set_description(f"Testing: ")

        scores = th.cat(scores, dim=0)[..., -1]
        # lst_layer = th.cat(lst_layer, dim=0)[..., -1]
        all_x = th.cat(all_x, dim=0)[..., -1]
        if self.debug:
            # save model
            th.save(self.model.state_dict(), "model.pth")
            # np.save("lst_layer.npy", lst_layer.numpy())
            np.save("all_x.npy", all_x.numpy())
        scores = scores.numpy().flatten()
        # Replace NaN in socres with 1000
        scores[np.isnan(scores)] = 1000
        assert scores.ndim == 1
        self.__anomaly_score = scores

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score  # type: ignore

    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(
            self.model, (self.batch_size, self.window), verbose=0
        )
        with open(save_file, "w") as f:
            f.write(str(model_stats))
        # pass
        # breakpoint()
