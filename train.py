from utils import rzbDataset, Tokenizer, Embedding
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import models


def main():
    # k-fold dataset
    model_name = "LSTM"
    batch_size = 16
    num_workers = 4
    epoch = 10
    learning_rate = 0.001
    model = load_model(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    test_dataset = rzbDataset("data", mode="test")
    test_dataloader = DataLoader(
        test_dataset, batch_size, shuffle=True, num_workers=num_workers)

    for k in range(9):
        train_dataset = rzbDataset("data", k, mode="train", method=pre_process)
        val_dataset = rzbDataset("data", k, mode="val")

        train_dataloader = DataLoader(
            train_dataset, batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(
            val_dataset, batch_size, shuffle=True, num_workers=num_workers)

        train(train_dataloader, epoch, model, batch_size,
              val_dataloader, optimizer, criterion)


def pre_process(text):
    tokenizer = Tokenizer()
    token = tokenizer.encode_as_pieces(text)
    token = Embedding().encode(token)
    return token


def load_model(model_name):
    model = getattr(models, model_name)
    model = getattr(model, model_name)()

    return model


def train(dataloader, epoch, model, batch_size, validation_data, optimizer, criterion):
    for i in range(epoch):
        model.train()  # 设置模型为训练模式

        total_loss = 0  # 用于记录每个epoch的总损失

        for step, (batch_x, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()  # 清空梯度

            outputs = model(batch_x)  # 前向传播
            loss = criterion(outputs, batch_y)  # 计算损失

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item()  # 累加损失

            if (step + 1) % 10 == 0:  # 每10个batch打印一次损失
                print(
                    f'Epoch [{i + 1}/{epoch}], Step [{step + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)  # 计算平均损失
        print(f'Epoch [{i + 1}/{epoch}], Average Loss: {avg_loss:.4f}')

        # 验证模型
        validate(model, validation_data, criterion)


def validate(model, validation_data, criterion):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        total_val_loss = 0

        for val_x, val_y in validation_data:
            outputs = model(val_x)  # 前向传播
            loss = criterion(outputs, val_y)  # 计算验证损失
            total_val_loss += loss.item()  # 累加验证损失

        avg_val_loss = total_val_loss / len(validation_data)  # 平均验证损失
        print(f'Validation Loss: {avg_val_loss:.4f}')


if __name__ == "__main__":
    main()
