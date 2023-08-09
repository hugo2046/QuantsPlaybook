"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-07-20 08:52:33
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-08-03 16:06:52
Description: 
"""
import torch
import torch.nn as nn
from .utils import reduce_dimensions
from typing import Tuple
from torch.utils.data import Dataset


#################################################################################
#                    构造数据集
#################################################################################


class CustomDataset(Dataset):
    def __init__(self, dataset: Tuple):
        self.features, self.next_returns, self.auxiliary = dataset
        feature_size: int = self.features.shape[0]
        auxiliary_size: int = self.auxiliary.shape[0]
        next_ret_size: int = self.next_returns.shape[0]
        if feature_size == auxiliary_size == next_ret_size:
            self.size = feature_size
        else:
            raise ValueError(
                f"The size of features({feature_size}), auxiliary({auxiliary_size}) and next_ret({next_ret_size}) are not equal."
            )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.features[index], self.next_returns[index], self.auxiliary[index]


#################################################################################
#                          任务模型
#################################################################################


class MainTaskNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_dim: int = 1,
    ):
        if num_layers < 1:
            raise ValueError("num_layers must be greater than 0")
        super(MainTaskNN, self).__init__()

        self.layers = nn.ModuleList()  # 用于存储隐藏层的列表

        # 添加输入层
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())

        # 添加动态数量的隐藏层
        if num_layers > 1:
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout))
        # 添加输出层，激活函数改为 Softmax
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        # 由于A股限制做空,预测结果在0-1之间且合计为1,因此使用Softmax作为激活函数
        # 设置dim=1保证在每一行上进行Softmax操作
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AuxTaskNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_dim: int = 1,
    ):
        if num_layers < 1:
            raise ValueError("num_layers must be greater than 0")
        super(AuxTaskNN, self).__init__()

        self.layers = nn.ModuleList()  # 用于存储隐藏层的列表

        # 添加输入层
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())

        # 添加动态数量的隐藏层
        if num_layers > 1:
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout))
        # 添加输出层
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# LSTM编码器
class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ):
        super(LSTMEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_dim).cuda()
        c0 = torch.randn(self.num_layers, batch_size, self.hidden_dim).cuda()
        output, _ = self.lstm(x, (h0, c0))
        return output


class Multi_Task_Model(nn.Module):
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int,
        mlp_hidden_size: int,
        lstm_layers: int = 1,
        mlp_layers: int = 1,
        lstm_dropout: float = None,
        mlp_dropout: float = 0,
        output: int = 1,
    ) -> None:
        super(Multi_Task_Model, self).__init__()

        # self.input_size = input_size
        # self.hidden_size = lstm_hidden_size
        # self.mpl_hidden_size = mpl_hidden_size
        # self.output = output

        self.lstm = LSTMEncoder(
            input_size, lstm_hidden_size, dropout=lstm_dropout, num_layers=lstm_layers
        )
        self.ffn_ctc = AuxTaskNN(
            lstm_hidden_size, mlp_hidden_size, mlp_layers, mlp_dropout, output
        )
        self.ffn_p = AuxTaskNN(
            lstm_hidden_size, mlp_hidden_size, mlp_layers, mlp_dropout, output
        )
        self.ffn_gk = AuxTaskNN(
            lstm_hidden_size, mlp_hidden_size, mlp_layers, mlp_dropout, output
        )
        self.ffn_rs = AuxTaskNN(
            lstm_hidden_size, mlp_hidden_size, mlp_layers, mlp_dropout, output
        )
        self.ffn_yz = AuxTaskNN(
            lstm_hidden_size, mlp_hidden_size, mlp_layers, mlp_dropout, output
        )
        self.main = MainTaskNN(
            lstm_hidden_size, mlp_hidden_size, mlp_layers, mlp_dropout, output
        )

    def forward(self, x):
        lstm_out: torch.Tensor = self.lstm(x)
        sigma_ctc: torch.Tensor = self.ffn_ctc(lstm_out)
        sigma_p: torch.Tensor = self.ffn_p(lstm_out)
        sigma_gk: torch.Tensor = self.ffn_gk(lstm_out)
        sigma_rs: torch.Tensor = self.ffn_rs(lstm_out)
        sigma_yz: torch.Tensor = self.ffn_yz(lstm_out)
        weight: torch.Tensor = self.main(lstm_out)

        sigma_matrix: torch.Tensor = torch.stack(
            tuple(
                reduce_dimensions(arr)
                for arr in (sigma_ctc, sigma_p, sigma_gk, sigma_rs, sigma_yz)
            ),
            dim=2,
        )

        return (sigma_matrix, reduce_dimensions(weight))
