#!/usr/bin/env python3
"""
预训练脚本 (Warm-up)
1. 使用 HeuristicAI 生成模拟对局数据
2. 使用这些数据对 PolicyValueNet 进行监督学习训练
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.connect6_env import Connect6Env, Player
from game.heuristic_ai import HeuristicAI
from model.network import (
    PolicyValueNet, PolicyValueNetSmall, PolicyValueNetLarge,
    PolicyValueNetAdvanced, PolicyValueNetHybrid
)
from training.trainer import GameDataset, augment_data

def create_network(model_type: str):
    """根据参数创建网络"""
    if model_type == 'small':
        return PolicyValueNetSmall()
    elif model_type == 'large':
        return PolicyValueNetLarge()
    elif model_type == 'advanced':
        return PolicyValueNetAdvanced(
            num_channels=192,
            num_res_blocks=12,
            num_transformer_blocks=2,
            num_heads=4,
            dropout_rate=0.1,
        )
    elif model_type == 'hybrid':
        return PolicyValueNetHybrid(
            base_channels=128,
            dropout_rate=0.1,
        )
    else:  # medium (default)
        return PolicyValueNet(
            num_channels=128,
            num_res_blocks=10,
            use_se=True,
            dropout_rate=0.1,
        )

def generate_data(num_games=1000):
    """生成预训练数据"""
    print(f"正在生成 {num_games} 局预训练数据...")
    ai = HeuristicAI(noise_prob=0.15) # 稍微增加一点随机性以覆盖更多局面
    env = Connect6Env()

    data_buffer = []

    for i in tqdm(range(num_games)):
        env.reset()
        game_record = [] # (state, policy, player)

        while not env.game_over:
            state = env.get_state()
            action = ai.get_action(env)

            # 构建 one-hot policy
            policy = np.zeros(361, dtype=np.float32)
            policy[action] = 1.0

            game_record.append((state, policy, env.current_player))
            env.step(action)

        # 游戏结束，回填价值
        winner = env.winner
        for state, policy, player in game_record:
            if winner is None:
                value = 0.0
            else:
                value = 1.0 if player == winner else -1.0

            # 数据增强 (旋转/翻转)
            augmented = augment_data(state, policy, value)
            data_buffer.extend(augmented)

    print(f"生成完成! 共收集 {len(data_buffer)} 个样本 (含增强).")
    return data_buffer

def train_supervised(network, data, epochs=5, batch_size=512, device='cuda'):
    """监督学习训练"""
    print(f"开始监督学习训练 (Epochs: {epochs}, Batch: {batch_size})...")
    network.to(device)
    network.train()

    # 解包数据
    states = np.array([x[0] for x in data])
    policies = np.array([x[1] for x in data])
    values = np.array([x[2] for x in data])

    dataset = GameDataset(states, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(network.parameters(), lr=0.001, weight_decay=1e-4)
    criterion_value = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        p_loss_sum = 0
        v_loss_sum = 0
        batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for b_states, b_policies, b_values in pbar:
            b_states = b_states.to(device)
            b_policies = b_policies.to(device)
            b_values = b_values.to(device).unsqueeze(1)

            optimizer.zero_grad()

            pred_policies, pred_values = network(b_states)

            # 策略损失 (Cross Entropy)
            # 注意：b_policies 是 one-hot 的，所以直接用 log_softmax + sum
            log_probs = torch.nn.functional.log_softmax(pred_policies, dim=1)
            policy_loss = -torch.sum(b_policies * log_probs) / b_states.size(0)

            # 价值损失 (MSE)
            value_loss = criterion_value(pred_values, b_values)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            p_loss_sum += policy_loss.item()
            v_loss_sum += value_loss.item()
            batches += 1

            pbar.set_postfix({
                'loss': total_loss/batches,
                'p_loss': p_loss_sum/batches,
                'v_loss': v_loss_sum/batches
            })

    return network

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='六子棋预训练脚本')
    parser.add_argument('--model-size', type=str, default='medium',
                        choices=['small', 'medium', 'large', 'advanced', 'hybrid'],
                        help='模型类型')
    parser.add_argument('--num-games', type=int, default=500,
                        help='生成对局数')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='批次大小')

    args = parser.parse_args()

    # 配置
    NUM_GAMES = args.num_games
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_PATH = "checkpoints_5/warmup_model.pt"

    os.makedirs("checkpoints", exist_ok=True)

    print(f"预训练配置: 模型={args.model_size}, 对局={NUM_GAMES}, Epochs={EPOCHS}, 设备={DEVICE}")

    # 1. 生成数据
    raw_data = generate_data(NUM_GAMES)

    # 2. 初始化网络
    print(f"初始化网络 ({args.model_size})...")
    net = create_network(args.model_size)

    # 3. 训练
    net = train_supervised(net, raw_data, epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE)

    # 4. 保存
    torch.save({
        'model_state_dict': net.state_dict(),
        'model_type': args.model_size
    }, SAVE_PATH)
    print(f"预训练模型已保存至: {SAVE_PATH}")
    print("现在你可以运行: python train.py --resume checkpoints_4/warmup_model.pt")
