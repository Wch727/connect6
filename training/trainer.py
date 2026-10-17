"""
AlphaZero训练器
实现自我对弈、数据收集和神经网络训练
"""

import os
import sys
import time
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from collections import deque
import pickle
from datetime import datetime
import multiprocessing as mp
# 并行自我对弈工具使用multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.connect6_env import Connect6Env, Player
from model.network import PolicyValueNet
from mcts.mcts import MCTS, MCTSParallel


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, max_size: int = 500000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state: np.ndarray, policy: np.ndarray, value: float):
        """添加一条经验"""
        self.buffer.append((state, policy, value))

    def push_batch(self, data: List[Tuple[np.ndarray, np.ndarray, float]]):
        """批量添加经验"""
        for item in data:
            self.buffer.append(item)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """随机采样一批数据"""
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)

        states = np.array([self.buffer[i][0] for i in indices])
        policies = np.array([self.buffer[i][1] for i in indices])
        values = np.array([self.buffer[i][2] for i in indices])

        return states, policies, values

    def __len__(self):
        return len(self.buffer)

    def save(self, path: str):
        """保存缓冲区"""
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path: str):
        """加载缓冲区"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.buffer = deque(data, maxlen=self.buffer.maxlen)
        except (EOFError, pickle.UnpicklingError) as exc:
            # 如果缓冲文件为空或损坏，跳过加载以免中断训练
            self.buffer.clear()
            print(f"警告: 经验回放文件无法读取({exc})，将从空缓冲区继续训练")


class GameDataset(Dataset):
    """游戏数据集"""

    def __init__(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray):
        self.states = torch.FloatTensor(states)
        self.policies = torch.FloatTensor(policies)
        self.values = torch.FloatTensor(values)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


def augment_data(state: np.ndarray, policy: np.ndarray, value: float) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    数据增强：通过旋转和翻转生成8个等价样本
    这对于棋类游戏非常重要，可以有效增加训练数据
    """
    augmented = []
    board_size = state.shape[1]

    # 将策略从1D转换为2D
    policy_2d = policy.reshape(board_size, board_size)

    for k in range(4):  # 4种旋转
        # 旋转状态和策略
        rotated_state = np.rot90(state, k, axes=(1, 2)).copy()
        rotated_policy = np.rot90(policy_2d, k).flatten().copy()
        augmented.append((rotated_state, rotated_policy, value))

        # 水平翻转
        flipped_state = np.flip(rotated_state, axis=2).copy()
        flipped_policy = np.flip(np.rot90(policy_2d, k), axis=1).flatten().copy()
        augmented.append((flipped_state, flipped_policy, value))

    return augmented


def _self_play_worker_entry(args: Dict[str, Any]) -> Dict[str, Any]:
    """多进程自我对弈工作函数"""
    import torch

    worker_id = args['worker_id']
    num_games = args['num_games']
    network_class = args['network_class']
    network_kwargs = args.get('network_kwargs') or {}
    state_dict = args['state_dict']
    worker_conf = args['worker_conf']
    device = args['device']
    seed = args.get('seed')

    torch.set_num_threads(1)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    if device.startswith('cuda') and torch.cuda.is_available():
        target_device = device
    else:
        target_device = 'cpu'

    network = network_class(**network_kwargs)
    network.load_state_dict(state_dict)
    network = network.to(target_device)

    worker = SelfPlayWorker(
        network=network,
        num_simulations=worker_conf['num_simulations'],
        c_puct=worker_conf['c_puct'],
        temperature=worker_conf['temperature'],
        num_parallel=worker_conf['num_parallel'],
        forced_random_moves=worker_conf.get('forced_random_moves', 0),
        use_rule_based=worker_conf.get('use_rule_based', True),
    )

    games: List[List[Tuple[np.ndarray, np.ndarray, float]]] = []
    durations: List[float] = []
    lengths: List[int] = []

    for _ in range(num_games):
        start_time = time.time()
        game_data = worker.play_game(add_noise=True)
        durations.append(time.time() - start_time)
        lengths.append(len(game_data))
        games.append(game_data)

    # 释放GPU显存
    del network

    return {
        'worker_id': worker_id,
        'games': games,
        'durations': durations,
        'lengths': lengths,
    }


class SelfPlayWorker:
    """自我对弈工作器 - 增强版"""

    def __init__(
        self,
        network: PolicyValueNet,
        num_simulations: int = 400,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        temperature_threshold: int = 30,  # 30步后降低温度
        use_progressive_temperature: bool = True,  # 渐进式温度
        num_parallel: int = 1,  # 并行搜索数
        forced_random_moves: int = 0,  # 强制随机步数
        use_rule_based: bool = True,  # 是否使用规则辅助
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.use_progressive_temperature = use_progressive_temperature
        self.num_parallel = num_parallel
        self.forced_random_moves = forced_random_moves
        self.use_rule_based = use_rule_based

    def _get_temperature(self, move_count: int) -> float:
        """获取当前步数对应的温度（渐进式降低）"""
        if not self.use_progressive_temperature:
            return self.temperature if move_count < self.temperature_threshold else 0.1

        # 渐进式温度：从1.0逐渐降低到0.1
        if move_count < 10:
            return self.temperature  # 前10步保持高温度鼓励探索
        elif move_count < self.temperature_threshold:
            # 线性降低
            progress = (move_count - 10) / (self.temperature_threshold - 10)
            return self.temperature * (1 - progress * 0.7)  # 降到0.3
        else:
            return 0.1  # 后期低温度

    def play_game(self, add_noise: bool = True) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        进行一局自我对弈

        Returns:
            game_data: [(state, policy, value), ...] 游戏数据
        """
        env = Connect6Env()

        # 混合训练策略：
        # 50% 的概率开启规则辅助（学习正确策略，Teacher Forcing）
        # 50% 的概率关闭规则辅助（探索失败边界，修正 Value Network，防止对规则产生依赖）
        # 这样模型既能学会"怎么赢"，也能学会"为什么不防守会输"
        use_rules_this_game = self.use_rule_based
        # if self.use_rule_based:
        if self.use_rule_based and random.random() < 0.01:
            use_rules_this_game = False

        if self.num_parallel > 1:
            mcts = MCTSParallel(
                self.network,
                c_puct=self.c_puct,
                num_simulations=self.num_simulations,
                num_parallel=self.num_parallel,
                use_rule_based=use_rules_this_game,
            )
        else:
            mcts = MCTS(
                self.network,
                c_puct=self.c_puct,
                num_simulations=self.num_simulations,
                use_rule_based=use_rules_this_game,
            )

        game_data = []
        move_count = 0
        rule_moves = 0

        while not env.game_over:
            # 渐进式温度调整
            temp = self._get_temperature(move_count)

            # 统计规则触发情况 (仅当本局开启规则时)
            if use_rules_this_game:
                w, d = env.get_urgent_moves()
                if w or d:
                    rule_moves += 1

            # 获取当前状态
            state = env.get_state()

            # MCTS搜索（前期加噪声，后期不加）
            use_noise = add_noise and move_count < self.temperature_threshold

            # 强制随机开局
            if move_count < self.forced_random_moves:
                # 随机策略 - 集中在中间且不连成6子
                valid_moves = env.get_valid_moves()
                candidates = []

                # 定义中间区域 (19x19棋盘，取中间11x11区域)
                center_start = 4
                center_end = 14

                # 第一次筛选：中间区域且不赢
                for action in np.where(valid_moves > 0)[0]:
                    x, y = env.action_to_coord(action)

                    # 检查是否在中间区域
                    if not (center_start <= x <= center_end and center_start <= y <= center_end):
                        continue

                    # 检查是否会赢 (模拟落子)
                    env.board[x, y] = env.current_player
                    is_win = env._check_win(x, y)
                    env.board[x, y] = Player.EMPTY

                    if not is_win:
                        candidates.append(action)

                # 如果中间区域没有合适步数（极少见），尝试全盘不赢的步数
                if not candidates:
                    for action in np.where(valid_moves > 0)[0]:
                        x, y = env.action_to_coord(action)

                        # 检查是否会赢
                        env.board[x, y] = env.current_player
                        is_win = env._check_win(x, y)
                        env.board[x, y] = Player.EMPTY

                        if not is_win:
                            candidates.append(action)

                # 如果还是没有（说明必赢），就随机选一个合法的
                if not candidates:
                    candidates = np.where(valid_moves > 0)[0].tolist()

                # 构建策略
                policy = np.zeros_like(valid_moves)
                for action in candidates:
                    policy[action] = 1.0 / len(candidates)
                # 不进行MCTS搜索，直接使用均匀分布
            elif self.num_parallel > 1:
                policy = mcts.search(env, add_noise=use_noise, temperature=temp)
            else:
                mcts.temperature = temp
                policy = mcts.search(env, add_noise=use_noise)

            # 保存数据（不包括最终价值，后面统一填充）
            # 只有当不是强制随机步数时才保存数据，避免模型学习到随机策略
            if move_count >= self.forced_random_moves:
                game_data.append((state.copy(), policy.copy(), env.current_player))

            # 选择动作
            if move_count < self.forced_random_moves:
                action = np.random.choice(len(policy), p=policy)
            elif temp > 0.1:
                action = np.random.choice(len(policy), p=policy)
            else:
                action = np.argmax(policy)

            # 执行动作
            env.step(action)
            move_count += 1

        # 根据游戏结果填充价值（游戏结束后执行一次）
        if env.winner is None:
            # 平局
            final_data = [(s, p, 0.0) for s, p, _ in game_data]
        else:
            # 有胜者
            final_data = []
            for state, policy, player in game_data:
                if player == env.winner:
                    value = 1.0
                else:
                    value = -1.0
                final_data.append((state, policy, value))

        # 打印规则触发统计 (仅在有规则触发时打印，避免刷屏)
        if use_rules_this_game:
            if rule_moves > 0:
                print(f"Game finished (Rules ON). Rule-based moves: {rule_moves}/{move_count} ({rule_moves/move_count:.1%})")
            else:
                print(f"Game finished (Rules ON). No rule-based moves triggered. Total moves: {move_count}")

        return final_data


class AlphaZeroTrainer:
    """AlphaZero训练器 - 增强版"""

    def __init__(
        self,
        network: PolicyValueNet,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 0.002,  # 提高初始学习率
        lr_min: float = 1e-5,  # 最小学习率
        weight_decay: float = 1e-4,
        batch_size: int = 512,  # 增大batch size
        num_simulations: int = 400,
        num_self_play_games: int = 100,
        num_epochs: int = 10,  # 增加epoch数
        buffer_size: int = 500000,
        checkpoint_dir: str = 'checkpoints',
        c_puct: float = 1.5,
        temperature: float = 1.0,
        use_data_augmentation: bool = True,
        # 新增参数
        entropy_weight: float = 0.01,  # 策略熵正则化权重
        value_loss_weight: float = 1.0,  # 价值损失权重
        kl_target: float = 0.02,  # 目标KL散度
        lr_multiplier: float = 1.0,  # 学习率乘数
        use_priority_sampling: bool = True,  # 优先级采样
        num_parallel: int = 1,  # 并行搜索数
        num_self_play_workers: int = 1,  # 自我对弈进程数
        self_play_device: Optional[str] = None,  # 自我对弈设备
        seed: Optional[int] = None,  # 随机种子
        forced_random_moves: int = 6,  # 强制随机步数 (默认6步，即双方各下3手)
        use_rule_based: bool = True,  # 是否使用规则辅助
    ):
        self.network = network.to(device)
        self.network_class = self.network.__class__
        self.network_kwargs = getattr(self.network, 'init_kwargs', {})
        self.device = device
        self.lr = lr
        self.lr_min = lr_min
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_simulations = num_simulations
        self.num_self_play_games = num_self_play_games
        self.num_epochs = num_epochs
        self.c_puct = c_puct
        self.temperature = temperature
        self.use_data_augmentation = use_data_augmentation
        self.num_parallel = num_parallel
        self.forced_random_moves = forced_random_moves
        self.use_rule_based = use_rule_based

        # 新增参数
        self.entropy_weight = entropy_weight
        self.value_loss_weight = value_loss_weight
        self.kl_target = kl_target
        self.lr_multiplier = lr_multiplier
        self.use_priority_sampling = use_priority_sampling
        self.num_self_play_workers = max(1, num_self_play_workers)
        self.seed = seed
        if self_play_device is None:
            default_sp_device = device if self.num_self_play_workers == 1 else 'cpu'
            self.self_play_device = default_sp_device
        else:
            self.self_play_device = self_play_device
        if self.self_play_device == 'cuda' and not torch.cuda.is_available():
            self.self_play_device = 'cpu'

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 使用AdamW优化器（更好的权重衰减）
        self.optimizer = optim.AdamW(
            network.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # 使用余弦退火学习率调度器（带热重启）
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,  # 首次重启周期
            T_mult=2,  # 重启周期倍数
            eta_min=lr_min
        )

        self.current_base_lrs: List[float] = []
        self._set_optimizer_base_lrs([self.lr for _ in self.optimizer.param_groups])

        # 训练统计
        self.iteration = 0
        self.total_games = 0
        self.train_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'entropy': [],  # 策略熵
            'kl_divergence': [],  # KL散度
            'learning_rate': [],  # 学习率
        }

        # 记录旧策略用于计算KL散度
        self.old_policies = None

    def _process_game_data(self, game_data: List[Tuple[np.ndarray, np.ndarray, float]]) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], int]:
        """统一处理自我对弈样本（可选数据增强）"""
        original_length = len(game_data)
        if not self.use_data_augmentation:
            return game_data, original_length
        augmented_data: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for state, policy, value in game_data:
            augmented_data.extend(augment_data(state, policy, value))
        return augmented_data, original_length

    def _set_optimizer_base_lrs(self, base_lrs: List[float]):
        """记录调度器给出的基础学习率并应用当前乘数"""
        self.current_base_lrs = base_lrs
        self._apply_lr_multiplier()

    def _apply_lr_multiplier(self):
        if not self.current_base_lrs:
            self.current_base_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
        for param_group, base_lr in zip(self.optimizer.param_groups, self.current_base_lrs):
            param_group['lr'] = base_lr * self.lr_multiplier

    def self_play(self, num_games: int = None) -> int:
        """
        执行自我对弈收集数据

        Returns:
            收集的样本数
        """
        if num_games is None:
            num_games = self.num_self_play_games

        if self.num_self_play_workers <= 1:
            return self._self_play_single(num_games)
        return self._self_play_parallel(num_games)

    def _self_play_single(self, num_games: int) -> int:
        self.network.eval()
        worker = SelfPlayWorker(
            self.network,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            temperature=self.temperature,
            num_parallel=self.num_parallel,
            forced_random_moves=self.forced_random_moves,
            use_rule_based=self.use_rule_based,
        )
        total_samples = 0
        for game_idx in range(num_games):
            start_time = time.time()
            game_data = worker.play_game(add_noise=True)
            duration = time.time() - start_time
            processed_data, game_length = self._process_game_data(game_data)
            self.replay_buffer.push_batch(processed_data)
            total_samples += len(processed_data)
            self.total_games += 1
            step_time = duration / max(game_length, 1)
            print(
                f"  第 {game_idx + 1}/{num_games} 局: "
                f"{duration:.1f}s, {game_length}步 ({step_time:.2f}s/步)"
            )
        return total_samples

    def _self_play_parallel(self, num_games: int) -> int:
        self.network.eval()
        state_dict_cpu = {k: v.detach().cpu() for k, v in self.network.state_dict().items()}
        worker_conf = {
            'num_simulations': self.num_simulations,
            'c_puct': self.c_puct,
            'temperature': self.temperature,
            'num_parallel': self.num_parallel,
            'forced_random_moves': self.forced_random_moves,
            'use_rule_based': self.use_rule_based,
        }

        # 将任务拆分为单局，以便实时监控进度
        # 虽然这会增加一些模型初始化的开销，但能提供更好的用户体验
        tasks: List[Dict[str, Any]] = []
        for i in range(num_games):
            worker_seed = (self.seed + i) if self.seed is not None else None
            tasks.append({
                'worker_id': i,
                'num_games': 1, # 每个任务只跑1局
                'network_class': self.network_class,
                'network_kwargs': self.network_kwargs,
                'state_dict': state_dict_cpu,
                'worker_conf': worker_conf,
                'device': self.self_play_device,
                'seed': worker_seed,
            })

        print(f"  启动 {self.num_self_play_workers} 个进程进行并行自我对弈 (共 {num_games} 局)")
        print(f"      (设备: {self.self_play_device}, MCTS并行: {self.num_parallel})")
        if self.self_play_device == 'cpu':
            print("      注意: 使用CPU进行自我对弈可能会比较慢，建议在显存允许的情况下使用 --self-play-device cuda")

        ctx = mp.get_context('spawn')
        total_samples = 0
        durations: List[float] = []
        lengths: List[int] = []

        start_time = time.time()
        completed_games = 0

        with ctx.Pool(processes=self.num_self_play_workers) as pool:
            # 使用 imap_unordered 获取实时结果
            for res in pool.imap_unordered(_self_play_worker_entry, tasks):
                completed_games += 1

                durations.extend(res['durations'])
                lengths.extend(res['lengths'])
                for game_data in res['games']:
                    processed_data, game_length = self._process_game_data(game_data)
                    self.replay_buffer.push_batch(processed_data)
                    total_samples += len(processed_data)
                    self.total_games += 1

                # 计算统计信息
                elapsed = time.time() - start_time
                avg_time = elapsed / completed_games
                remaining = num_games - completed_games
                eta = avg_time * remaining

                last_len = lengths[-1] if lengths else 0

                # 动态打印进度
                print(f"\r  进度: {completed_games}/{num_games} | "
                      f"耗时: {elapsed:.0f}s | "
                      f"ETA: {eta:.0f}s | "
                      f"最新步数: {last_len} ", end="", flush=True)

        print() # 换行

        games_played = len(lengths)
        if games_played > 0:
            avg_duration = sum(durations) / games_played
            avg_length = sum(lengths) / games_played
            avg_step = avg_duration / max(avg_length, 1)
            print(
                f"  完成: {games_played} 局 | "
                f"均时: {avg_duration:.1f}s | "
                f"均步: {avg_length:.1f} | "
                f"速度: {avg_step:.2f}s/步"
            )
        return total_samples

    def train_network(self, num_epochs: int = None) -> Dict[str, float]:
        """
        训练神经网络 - 增强版
        包含：KL散度监控、策略熵正则化、动态学习率调整

        Returns:
            训练损失统计
        """
        if num_epochs is None:
            num_epochs = self.num_epochs

        if len(self.replay_buffer) < self.batch_size:
            print("缓冲区数据不足，跳过训练")
            return {}

        self.network.train()

        # 采样训练数据 - 增加采样量
        sample_size = min(len(self.replay_buffer), self.batch_size * num_epochs * 10)
        states, policies, values = self.replay_buffer.sample(sample_size)

        # 保存旧策略用于计算KL散度
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states[:min(1024, len(states))]).to(self.device)
            old_policy_logits, _ = self.network(states_tensor)
            old_probs = F.softmax(old_policy_logits, dim=1)
            self.old_policies = old_probs.cpu().numpy()

        dataset = GameDataset(states, policies, values)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False,
            drop_last=True  # 丢弃不完整的batch
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0
        epoch_kl = 0.0  # 在循环外初始化，确保变量始终存在

        for epoch in range(num_epochs):
            epoch_batches = 0

            for batch_states, batch_policies, batch_values in dataloader:
                batch_states = batch_states.to(self.device)
                batch_policies = batch_policies.to(self.device)
                batch_values = batch_values.to(self.device).unsqueeze(1)

                # 前向传播
                policy_logits, values_pred = self.network(batch_states)

                # 计算策略概率分布
                policy_probs = F.softmax(policy_logits, dim=1)
                log_probs = F.log_softmax(policy_logits, dim=1)

                # 1. 策略损失: 使用软标签的KL散度损失（比纯交叉熵更稳定）
                # 添加标签平滑
                smooth_policies = batch_policies * 0.99 + 0.01 / 361
                policy_loss = -(smooth_policies * log_probs).sum(dim=1).mean()

                # 2. 策略熵正则化（鼓励探索）
                entropy = -(policy_probs * log_probs).sum(dim=1).mean()

                # 3. 价值损失: Huber Loss（对异常值更鲁棒）
                value_loss = F.smooth_l1_loss(values_pred, batch_values)

                # 4. 总损失 = 策略损失 + 价值损失 - 熵正则化
                loss = (policy_loss +
                        self.value_loss_weight * value_loss -
                        self.entropy_weight * entropy)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度监控（检测异常梯度）
                grad_norm = 0.0
                for p in self.network.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)

                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_loss += loss.item()
                total_grad_norm += grad_norm
                num_batches += 1
                epoch_batches += 1

            # 每个epoch结束后计算KL散度
            if self.old_policies is not None and epoch_batches > 0:
                with torch.no_grad():
                    new_policy_logits, _ = self.network(states_tensor)
                    new_probs = F.softmax(new_policy_logits, dim=1).cpu().numpy()

                    # 计算KL散度
                    kl_div = np.mean(np.sum(
                        self.old_policies * (np.log(self.old_policies + 1e-10) -
                                             np.log(new_probs + 1e-10)),
                        axis=1
                    ))
                    epoch_kl = kl_div

                    # 动态调整学习率乘数基于KL散度
                    # 注意：lr_multiplier 会与调度器的学习率相乘
                    if kl_div > self.kl_target * 2:
                        # KL太大，减小学习率乘数
                        self.lr_multiplier = max(0.5, self.lr_multiplier / 1.5)
                    elif kl_div < self.kl_target / 2:
                        # KL太小，增大学习率乘数
                        self.lr_multiplier = min(2.0, self.lr_multiplier * 1.5)

            # 每个epoch结束后更新调度器的基础学习率
            self.scheduler.step()
            self._set_optimizer_base_lrs(self.scheduler.get_last_lr())
            # 应用KL散度调整后的学习率乘数（在调度器更新后应用）
            self._apply_lr_multiplier()

        # 记录统计
        avg_policy_loss = total_policy_loss / max(num_batches, 1)
        avg_value_loss = total_value_loss / max(num_batches, 1)
        avg_entropy = total_entropy / max(num_batches, 1)
        avg_total_loss = total_loss / max(num_batches, 1)
        avg_grad_norm = total_grad_norm / max(num_batches, 1)
        current_lr = self.optimizer.param_groups[0]['lr']

        self.train_history['policy_loss'].append(avg_policy_loss)
        self.train_history['value_loss'].append(avg_value_loss)
        self.train_history['total_loss'].append(avg_total_loss)
        self.train_history['entropy'].append(avg_entropy)
        self.train_history['kl_divergence'].append(epoch_kl)
        self.train_history['learning_rate'].append(current_lr)

        # 检测训练异常
        if avg_grad_norm > 10.0:
            print(f"  警告: 梯度范数过大 ({avg_grad_norm:.2f})，可能存在训练不稳定")
        if avg_policy_loss > 5.0:
            print(f"  警告: 策略损失过大 ({avg_policy_loss:.4f})，检查数据质量")

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'total_loss': avg_total_loss,
            'kl_divergence': epoch_kl if 'epoch_kl' in dir() else 0.0,
            'learning_rate': current_lr,
            'grad_norm': avg_grad_norm,
        }

    def train(self, num_iterations: int = 100, save_interval: int = 10):
        """
        主训练循环

        Args:
            num_iterations: 本次额外执行的迭代次数（恢复训练时接续已有编号）
            save_interval: 保存检查点间隔
        """
        print(f"\n{'='*60}")
        print(f"开始 AlphaZero 训练")
        print(f"{'='*60}")
        print(f"配置信息:")
        print(f"  • 设备: {self.device}")
        print(f"  • 迭代次数: {num_iterations}")
        print(f"  • 自我对弈: {self.num_self_play_games} 局/迭代")
        print(f"  • MCTS模拟: {self.num_simulations} 次")
        print(f"  • 批次大小: {self.batch_size}")
        print(f"  • 训练轮数: {self.num_epochs}")
        print(f"{'-'*60}\n")

        start_time = time.time()

        target_iteration = self.iteration + num_iterations
        for iteration in range(self.iteration + 1, target_iteration + 1):
            self.iteration = iteration
            iter_start = time.time()

            # 打印迭代标题
            print(f"迭代 {self.iteration}/{target_iteration} [{'='*20}]")

            # 自我对弈
            print(f"\n[1/3] 自我对弈收集数据 (Workers: {self.num_self_play_workers})...")
            sp_start = time.time()
            samples = self.self_play()
            sp_time = time.time() - sp_start

            # 训练网络
            print(f"\n[2/3] 训练神经网络 (Epochs: {self.num_epochs})...")
            train_start = time.time()
            losses = self.train_network()
            train_time = time.time() - train_start

            iter_time = time.time() - iter_start
            total_time = time.time() - start_time

            # 打印详细统计
            print(f"\n[3/3] 迭代 {self.iteration} 统计报告:")
            print(f"  {'-'*50}")
            print(f"  时间统计")
            print(f"    • 自我对弈: {sp_time:.1f}s")
            print(f"    • 网络训练: {train_time:.1f}s")
            print(f"    • 迭代总时: {iter_time:.1f}s")
            print(f"    • 累计运行: {total_time/60:.1f}min")
            print(f"  ")
            print(f"  数据统计")
            print(f"    • 新增样本: {samples}")
            print(f"    • 缓冲总量: {len(self.replay_buffer)}")
            print(f"  ")
            print(f"  损失指标")
            print(f"    • 策略损失: {losses.get('policy_loss', 0):.4f}")
            print(f"    • 价值损失: {losses.get('value_loss', 0):.4f}")
            print(f"    • 策略熵值: {losses.get('entropy', 0):.4f}")
            print(f"    • KL散度  : {losses.get('kl_divergence', 0):.6f}")
            print(f"    • 学习率  : {losses.get('learning_rate', 0):.6f}")
            print(f"  {'-'*50}")

            # 保存检查点
            if self.iteration % save_interval == 0:
                print(f"\n保存检查点...")
                self.save_checkpoint()

            print("\n") # 空行分隔

        print(f"\n{'='*60}")
        print(f"训练完成!")
        print(f"{'='*60}")
        self.save_checkpoint(final=True)

    def save_checkpoint(self, final: bool = False, model_type: str = None):
        """保存检查点"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 如果没有传入 model_type，尝试从 self.model_type 获取
        if model_type is None:
            model_type = getattr(self, 'model_type', None)

        if final:
            path = os.path.join(self.checkpoint_dir, f"final_model.pt")
        else:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_iter{self.iteration}.pt")

        torch.save({
            'iteration': self.iteration,
            'total_games': self.total_games,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'model_type': model_type,
            'num_simulations': self.num_simulations,
            'lr_multiplier': self.lr_multiplier,
        }, path)

        # 保存经验回放
        buffer_path = os.path.join(self.checkpoint_dir, "replay_buffer.pkl")
        self.replay_buffer.save(buffer_path)

        print(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.iteration = checkpoint.get('iteration', 0)
        self.total_games = checkpoint.get('total_games', 0)

        if 'model_state_dict' in checkpoint:
            self.network.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.train_history = checkpoint.get('train_history', {
            'policy_loss': [], 'value_loss': [], 'total_loss': [],
            'entropy': [], 'kl_divergence': [], 'learning_rate': []
        })

        checkpoint_model_type = checkpoint.get('model_type', None)
        if checkpoint_model_type is not None:
            self.model_type = checkpoint_model_type

        self.lr_multiplier = checkpoint.get('lr_multiplier', self.lr_multiplier)

        if 'scheduler_state_dict' in checkpoint:
            self._set_optimizer_base_lrs(self.scheduler.get_last_lr())

        # 尝试加载经验回放
        buffer_path = os.path.join(os.path.dirname(path), "replay_buffer.pkl")
        if os.path.exists(buffer_path):
            self.replay_buffer.load(buffer_path)

        print(f"检查点已加载: {path}")
        print(f"当前迭代: {self.iteration}, 总对局数: {self.total_games}")

    @staticmethod
    def peek_checkpoint(path: str) -> dict:
        """预览检查点信息，不加载模型权重

        返回检查点中的元数据，包括模型类型和训练参数
        """
        checkpoint = torch.load(path, map_location='cpu')
        return {
            'iteration': checkpoint.get('iteration', 0),
            'total_games': checkpoint.get('total_games', 0),
            'model_type': checkpoint.get('model_type', None),
            'num_simulations': checkpoint.get('num_simulations', None),
        }


def test_trainer():
    """测试训练器"""
    print("测试AlphaZero训练器...")

    from model.network import PolicyValueNetSmall

    # 使用小型网络进行测试
    net = PolicyValueNetSmall()

    trainer = AlphaZeroTrainer(
        net,
        num_simulations=50,  # 减少模拟次数加快测试
        num_self_play_games=2,  # 减少对局数
        num_epochs=2,
        batch_size=32,
    )

    # 测试一次迭代
    trainer.train(num_iterations=1)

    print("\n训练器测试完成!")


if __name__ == "__main__":
    test_trainer()
