import numpy as np
import torch
import torch.utils.data

from EasyTSAD.DataFactory import TSData


class UTSOneByOneDataset(torch.utils.data.Dataset):
    """
    The Dateset for one by one training and testing.
    """

    def __init__(
        self, tsData: TSData, phase: str, window_size: int, horizon: int = 1
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.horizon = horizon
        self.max = tsData.train.max()
        self.min = tsData.train.min()

        if phase == "train":
            (self.len,) = tsData.train.shape
            self.sample_num = max(self.len - self.window_size - self.horizon + 1, 0)
            X = torch.zeros((self.sample_num, self.window_size))
            Y = torch.zeros((self.sample_num, 1))
            data: np.ndarray = 2 * tsData.train - 1  # type: ignore

            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(data[i : i + self.window_size])
                Y[i, :] = torch.from_numpy(
                    np.array(data[i + self.window_size + self.horizon - 1])
                )

        elif phase == "valid":
            (self.len,) = tsData.valid.shape
            self.sample_num = max(self.len - self.window_size - self.horizon + 1, 0)
            X = torch.zeros((self.sample_num, self.window_size))
            Y = torch.zeros((self.sample_num, 1))
            data: np.ndarray = np.clip(2 * tsData.valid - 1, -1, 1)  # type: ignore

            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(data[i : i + self.window_size])
                Y[i, :] = torch.from_numpy(
                    np.array(data[i + self.window_size + self.horizon - 1])
                )

        elif phase == "test":
            (self.len,) = tsData.test.shape
            self.sample_num = max(self.len - self.window_size - self.horizon + 1, 0)
            X = torch.zeros((self.sample_num, self.window_size))
            Y = torch.zeros((self.sample_num, 1))
            data: np.ndarray = np.clip(2 * tsData.test - 1, -1, 1)  # type: ignore

            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(data[i : i + self.window_size])
                Y[i, :] = torch.from_numpy(
                    np.array(data[i + self.window_size + self.horizon - 1])
                )

        else:
            raise ValueError(
                'Arg "phase" in OneByOneDataset() must be one of "train", "valid", "test"'
            )

        self.samples, self.targets = X, Y

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index, :], self.targets[index, :]
