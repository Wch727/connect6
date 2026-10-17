#!/usr/bin/env python3
"""
六子棋AlphaZero训练主脚本

针对P100 GPU优化，预计训练8小时左右可获得较强的AI

使用方法:
    python train.py --help  # 查看所有参数
    python train.py  # 使用默认参数训练
    python train.py --resume checkpoints/checkpoint_iter10.pt  # 从检查点恢复
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import random

# 确保输出不被缓冲
sys.stdout.reconfigure(line_buffering=True)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.connect6_env import Connect6Env
from model.network import (
    PolicyValueNet, PolicyValueNetSmall, PolicyValueNetLarge,
    PolicyValueNetAdvanced, PolicyValueNetHybrid, get_network, count_parameters
)
from training.trainer import AlphaZeroTrainer


def safe_torch_load(path: str, map_location: str = 'cpu'):
    """在不同PyTorch版本间安全加载checkpoint"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 旧版PyTorch无weights_only参数
        return torch.load(path, map_location=map_location)


def detect_model_type_from_checkpoint(path: str) -> str:
    """从检查点的 state_dict 推断模型类型"""
    checkpoint = safe_torch_load(path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', {})
    keys = list(state_dict.keys())

    # 先检查检查点是否保存了 model_type
    saved_type = checkpoint.get('model_type', None)
    if saved_type:
        return saved_type

    # 根据 state_dict 的 key 来推断模型类型
    if any('transformer' in k for k in keys):
        if any('stage1' in k for k in keys):
            return 'hybrid'
        else:
            return 'advanced'
    elif any('res_blocks' in k for k in keys):
        # 检查是否有 SE 模块
        if any('se.' in k for k in keys):
            # 检查 res_blocks 数量来区分 small/medium/large
            res_block_indices = set()
            for k in keys:
                if 'res_blocks.' in k:
                    idx = k.split('res_blocks.')[1].split('.')[0]
                    res_block_indices.add(int(idx))
            num_blocks = len(res_block_indices)
            if num_blocks <= 5:
                return 'small'
            elif num_blocks <= 10:
                return 'medium'
            else:
                return 'large'
        else:
            return 'small'
    elif any('stage1' in k for k in keys):
        return 'hybrid'

    # 默认返回 medium
    return 'medium'


def set_seed(seed: int):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def main():
    parser = argparse.ArgumentParser(description='六子棋AlphaZero训练')

    # 模型参数
    parser.add_argument('--model-size', type=str, default='medium',
                        choices=['small', 'medium', 'large', 'advanced', 'hybrid'],
                        help='模型类型: small(~500K), medium(~2M), large(~10M), advanced(~5M,带Transformer), hybrid(~3M,混合架构)')

    # 训练参数
    parser.add_argument('--iterations', type=int, default=200,
                        help='本次额外训练轮数；恢复训练时接着检查点编号继续')
    parser.add_argument('--self-play-games', type=int, default=50,
                        help='每次迭代的自我对弈局数')
    parser.add_argument('--self-play-workers', type=int, default=1,
                        help='并行自我对弈进程数')
    parser.add_argument('--num-simulations', type=int, default=400,
                        help='MCTS模拟次数')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=5,
                        help='每次迭代的训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减')

    # MCTS参数
    parser.add_argument('--c-puct', type=float, default=2.5,
                        help='PUCT探索系数')
    parser.add_argument('--temperature', type=float, default=1.5,
                        help='动作选择温度')
    parser.add_argument('--parallel', type=int, default=1,
                        help='MCTS并行搜索线程数 (GPU加速，建议8-16)')
    parser.add_argument('--no-rule-based', action='store_true',
                        help='禁用规则辅助 (get_urgent_moves)')
    parser.add_argument('--self-play-device', type=str, default='auto',
                        help='自我对弈进程使用的设备: auto/cpu/cuda')

    # 数据参数
    parser.add_argument('--buffer-size', type=int, default=500000,
                        help='经验回放缓冲区大小')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='禁用数据增强')


    # 其他参数
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='保存检查点间隔')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='auto',
                        help='运行设备: auto, cuda, cpu')

    # 新增训练增强参数
    parser.add_argument('--entropy-weight', type=float, default=0.01,
                        help='策略熵正则化权重（鼓励探索）')
    parser.add_argument('--kl-target', type=float, default=0.02,
                        help='目标KL散度（用于动态学习率调整）')
    parser.add_argument('--lr-min', type=float, default=1e-5,
                        help='最小学习率')
    parser.add_argument('--forced-random-moves', type=int, default=0,
                        help='强制随机开局步数 (防止模型坍缩)')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("六子棋 AlphaZero 训练 - 增强版")
    print("=" * 60)
    print(f"设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 如果要恢复训练，先从检查点检测模型类型
    model_type = args.model_size
    if args.resume:
        if os.path.exists(args.resume):
            detected_type = detect_model_type_from_checkpoint(args.resume)
            if detected_type != args.model_size:
                print(f"\n注意: 检测到检查点使用的是 {detected_type} 模型，自动切换模型类型")
                model_type = detected_type
        else:
            print(f"警告: 检查点文件不存在: {args.resume}")
            sys.exit(1)

    # 创建网络
    print(f"\n模型类型: {model_type}")
    network = create_network(model_type)
    num_params = count_parameters(network)
    print(f"模型参数量: {num_params:,} ({num_params/1e6:.2f}M)")

    # 打印网络架构信息
    if model_type == 'advanced':
        print("  架构: ResNet + Transformer (全局注意力)")
        print("  特性: GC-Net, CBAM, 多头自注意力")
    elif model_type == 'hybrid':
        print("  架构: Bottleneck + SE + Transformer")
        print("  特性: 三阶段特征提取, 轻量Transformer")

    # 如果使用 advanced 模型，建议降低学习率
    if args.model_size == 'advanced' and args.lr > 0.0005:
        print(f"提示: Advanced 模型建议使用较低的学习率 (当前: {args.lr}, 建议: 0.0002)")
        # args.lr = 0.0002 # 可以选择自动调整，这里仅提示

    self_play_device = None if args.self_play_device.lower() == 'auto' else args.self_play_device

    # 创建训练器（使用增强版参数）
    trainer = AlphaZeroTrainer(
        network=network,
        device=device,
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_simulations=args.num_simulations,
        num_self_play_games=args.self_play_games,
        num_epochs=args.epochs,
        buffer_size=args.buffer_size,
        checkpoint_dir=args.checkpoint_dir,
        c_puct=args.c_puct,
        temperature=args.temperature,
        use_data_augmentation=not args.no_augmentation,
        entropy_weight=args.entropy_weight,
        kl_target=args.kl_target,
        use_rule_based=not args.no_rule_based,
        num_parallel=args.parallel,
        num_self_play_workers=args.self_play_workers,
        self_play_device=self_play_device,
        seed=args.seed,
        forced_random_moves=args.forced_random_moves,
    )

    # 保存当前使用的模型类型，供保存检查点时使用
    trainer.model_type = model_type

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 打印训练配置
    print(f"\n训练配置:")
    print(f"  迭代次数: {args.iterations}")
    print(f"  每迭代对弈局数: {args.self_play_games}")
    print(f"  MCTS模拟次数: {args.num_simulations}")
    print(f"  MCTS并行数: {args.parallel}")
    print(f"  自我对弈进程: {args.self_play_workers} (设备: {trainer.self_play_device})")
    print(f"  批次大小: {args.batch_size}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  学习率: {args.lr} (min: {args.lr_min})")
    print(f"  熵权重: {args.entropy_weight}")
    print(f"  KL目标: {args.kl_target}")
    print(f"  数据增强: {not args.no_augmentation}")
    print(f"  缓冲区大小: {args.buffer_size}")

    # 估算训练时间
    base_time_per_game = 90  # 串行每局约90秒
    if args.parallel > 1:
        # 并行加速估算 (16线程约加速9倍)
        speedup = max(1.0, args.parallel * 0.55)
        time_per_game = base_time_per_game / speedup
    else:
        time_per_game = base_time_per_game

    total_seconds = args.iterations * args.self_play_games * time_per_game
    print(f"\n预估训练时间: {total_seconds/3600:.1f} 小时")
    print("=" * 60)

    # 开始训练
    try:
        trainer.train(
            num_iterations=args.iterations,
            save_interval=args.save_interval
        )
    except KeyboardInterrupt:
        print("\n\n训练被中断，保存当前状态...")
        trainer.save_checkpoint(model_type=model_type)
        print("状态已保存，可以使用 --resume 恢复训练")


if __name__ == "__main__":
    main()
