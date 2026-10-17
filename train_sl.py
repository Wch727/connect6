import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import csv
import sys
import glob # 用于查找文件

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.network import PolicyValueNetHybrid, PolicyValueNet
from training.trainer import GameDataset

def train_supervised_learning(
    data_dir="data",  # 改为目录
    save_dir="checkpoints",
    log_file="logs/sl_log.csv",
    epochs=20,
    batch_size=512,
    lr=0.001
):
    # === 关键修改：加载目录下所有 .pkl 文件 ===
    print(f"📂 正在扫描 {data_dir} 下的数据文件...")
    files = glob.glob(os.path.join(data_dir, "*.pkl"))
    if not files:
        print("❌ 没有找到数据文件！请先运行 gen_data.py")
        return

    all_datasets = []
    total_samples = 0

    for pkl_file in files:
        print(f"   -> 加载 {pkl_file} ...")
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                states = [x[0] for x in data]
                policies = [x[1] for x in data]
                values = [x[2] for x in data]
                dataset = GameDataset(states, policies, values)
                all_datasets.append(dataset)
                total_samples += len(data)
        except Exception as e:
            print(f"   ⚠️ 跳过损坏的文件 {pkl_file}: {e}")

    if not all_datasets:
        print("❌ 数据加载失败")
        return

    print(f"✅ 数据加载完成！总样本数: {total_samples}")

    # 合并所有小数据集
    full_dataset = ConcatDataset(all_datasets)
    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    # ... 以下代码和之前一样 ...
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'total_loss', 'policy_loss', 'value_loss'])

    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = PolicyValueNetHybrid().to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    loss_fn_v = nn.MSELoss()

    network.train()
    print("🔥 开始训练...")

    for epoch in range(epochs):
        total_loss_sum = 0
        batches = 0
        p_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for s, p, v in p_bar:
            s, p, v = s.to(device), p.to(device), v.to(device).unsqueeze(1)
            p_logits, v_pred = network(s)

            log_probs = torch.log_softmax(p_logits, dim=1)
            loss_p = -torch.sum(p * log_probs) / s.size(0)
            loss_v = loss_fn_v(v_pred, v)
            loss = loss_p + loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_sum += loss.item()
            batches += 1
            p_bar.set_postfix({'L': loss.item()})

        # 记录日志
        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, total_loss_sum/batches, 0, 0])

        # 保存中间模型
        if (epoch + 1) % 5 == 0:
            torch.save({'model_state_dict': network.state_dict()}, os.path.join(save_dir, f"sl_model_epoch_{epoch+1}.pt"))

    torch.save({'model_state_dict': network.state_dict()}, os.path.join(save_dir, "sl_model.pt"))
    print("✅ 预训练完成！")

if __name__ == "__main__":
    train_supervised_learning()
