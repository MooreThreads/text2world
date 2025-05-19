import torch
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from remap_pipeline import LatentAlign3D
import torch
import math
import torch.nn as nn
import os

# 数据预处理 (保持原始维度)


def prepare_data(datas, test_ratio=0.2):
    """返回划分好的训练集和测试集"""
    dlatents = torch.stack(
        [d['dlatent'].permute(0, 2, 1, 3, 4)[0] for d in datas])
    vlatents = torch.stack([d['vlatent'][0] for d in datas])

    assert dlatents.shape == vlatents.shape, \
        f"维度不匹配: dlatent {dlatents.shape} vs vlatent {vlatents.shape}"

    # 划分训练测试集
    dataset = TensorDataset(dlatents, vlatents)
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    return train_set, test_set

# 极简训练循环


def train_model(train_set, test_set, epochs=20, lr=1e-3, save_dir='checkpoints'):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    warmup_epochs = epochs*0.2
    # 自动获取输入维度
    C, V, W, H = train_set[0][0].shape
    print(f"训练数据维度: C={C}, V={V}, W={W}, H={H}")

    # 初始化模型和优化器
    model = LatentAlign3D(channels=C)
    model = model.to(torch.bfloat16)
    model = model.to('cuda')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    # scheduler = CosineWarmupLR(optimizer, warmup_epochs, epochs)
    # 数据加载
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    # 训练循环
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            train_loss += loss.item()

        # 测试集评估
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                output = model(x)
                test_loss += criterion(output, y).item()

        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss/len(train_loader):.4f} | '
              f'Test Loss: {test_loss/len(test_loader):.4f}')
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f'{save_dir}/best_model.pth')

    return model


def evaluate_model(model, test_set):
    test_loader = DataLoader(test_set, batch_size=32)
    criterion = nn.MSELoss()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            test_loss += criterion(output, y).item()

    print(f'\n最终测试集MSE: {test_loss/len(test_loader):.4f}')
    return test_loss/len(test_loader)


if __name__ == '__main__':
    datas = []
    for data_f in glob('*.torch'):
        datas.append(torch.load(data_f))
    train_set, test_set = prepare_data(datas, test_ratio=0.2)

    # 自动获取维度并训练
    model = train_model(train_set, test_set, lr=1e-3, epochs=5000)
