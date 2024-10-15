from mamba_ssm import Mamba as Mba
from utils.embedding import Embedding
import torch
import torch.nn as nn

# batch, length, dim = 2, 64, 16
# x = torch.randn(batch, length, dim).to("cuda")
# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim,  # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# y = model(x)

# print(x)
# print(y)


class Mamba(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.embedding = Embedding()
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded)
        output = self.fc(output)
        return output


# 创建一个输入序列
input_sequence = torch.tensor([[1, 2, 3, 4, 5]])

# 创建模型并进行前向传播
model = NLPModel(input_size=10, hidden_size=20, output_size=3)
print(model)
output_sequence = model(input_sequence)

print("输入序列的长度:", input_sequence.size(1))
print("输出序列的长度:", output_sequence.size(1))
