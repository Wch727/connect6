#!/usr/bin/env python3
"""
六子棋对弈接口
支持人机对弈和AI对战
"""

import argparse
import os
import sys
import numpy as np
import torch
import sys
import types

# ==============================================================================
# 🚑【V6 终极兼容版】伪造华为 NPU 环境 (针对 Windows/无NPU环境)
# 解决系列报错:
# 1. 模块/包结构缺失
# 2. _rebuild_npu_tensor 参数数量不匹配 (takes 6 but 7 given)
# 3. FakeFormat 参数接收问题
# ==============================================================================
try:
    import torch_npu
except ImportError:
    # 1. 伪造主包
    fake_npu = types.ModuleType('torch_npu')
    fake_npu.__path__ = []
    sys.modules['torch_npu'] = fake_npu

    # 2. 伪造 utils 子包
    fake_utils = types.ModuleType('torch_npu.utils')
    fake_utils.__path__ = []
    sys.modules['torch_npu.utils'] = fake_utils
    fake_npu.utils = fake_utils

    # 3. 伪造 storage 模块
    fake_storage = types.ModuleType('torch_npu.utils.storage')
    sys.modules['torch_npu.utils.storage'] = fake_storage
    fake_utils.storage = fake_storage

    # === 关键修改：增加 *args 接收多余的参数 ===
    def _rebuild_npu_tensor_mock(storage, storage_offset, size, stride, requires_grad, backward_hooks, *args):
        # 这里的 *args 会吃掉第 7 个参数 (NPU Format)，我们直接忽略它
        # 只把标准的 6 个参数传给 CPU 重建函数
        return torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks)

    fake_storage._rebuild_npu_tensor = _rebuild_npu_tensor_mock

    # 4. 伪造 npu 子包
    fake_npu_core = types.ModuleType('torch_npu.npu')
    fake_npu_core.__path__ = []
    sys.modules['torch_npu.npu'] = fake_npu_core
    fake_npu.npu = fake_npu_core

    # 5. 伪造 _format 模块 & Format 类
    fake_format = types.ModuleType('torch_npu.npu._format')
    sys.modules['torch_npu.npu._format'] = fake_format
    fake_npu_core._format = fake_format

    class FakeFormat:
        def __init__(self, *args, **kwargs):
            pass

    fake_format.Format = FakeFormat

    print("⚠️ 检测到无 NPU 环境，已启用 V6 全兼容伪造补丁...")
# ==============================================================================

# ... 下面接原本的代码 ...

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.connect6_env import Connect6Env, Player
from model.network import (
    PolicyValueNet, PolicyValueNetSmall, PolicyValueNetLarge,
    PolicyValueNetAdvanced, PolicyValueNetHybrid
)
from mcts.mcts import MCTS, MCTSParallel


def detect_model_type(state_dict):
    """根据state_dict的key自动检测模型类型"""
    keys = list(state_dict.keys())

    # hybrid模型有 stem, stage1, stage2, stage3, transformer
    if any('stem' in k for k in keys) and any('stage1' in k for k in keys):
        return 'hybrid'

    # advanced模型有 res_blocks 和 transformer_blocks
    if any('transformer_blocks' in k for k in keys):
        return 'advanced'

    # 其他是标准 PolicyValueNet 系列
    if 'input_conv.conv.weight' in state_dict:
        num_channels = state_dict['input_conv.conv.weight'].shape[0]
        # 计算残差块数量
        res_block_count = sum(1 for k in keys if k.startswith('res_blocks.') and k.endswith('.conv1.weight'))

        if num_channels <= 64:
            return 'small'
        elif num_channels >= 256 or res_block_count >= 20:
            return 'large'
        else:
            return 'medium'

    return 'medium'  # 默认


def create_network_by_type(model_type: str):
    """根据类型创建网络"""
    if model_type == 'small':
        return PolicyValueNetSmall()
    elif model_type == 'large':
        return PolicyValueNetLarge()
    elif model_type == 'advanced':
        return PolicyValueNetAdvanced()
    elif model_type == 'hybrid':
        return PolicyValueNetHybrid()
    else:  # medium
        return PolicyValueNet()


class Connect6Player:
    """六子棋AI玩家"""

    def __init__(
        self,
        model_path: str = None,
        num_simulations: int = 800,
        c_puct: float = 1.5,
        temperature: float = None,
        deterministic: bool = True,
        device: str = 'auto',
        model_type: str = None,  # 新增：手动指定模型类型
        num_parallel: int = 1,   # 新增：并行搜索数
        use_rule_based: bool = True,  # 新增：是否使用规则辅助
    ):
        """
        初始化AI玩家

        Args:
            model_path: 模型文件路径，如果为None则使用随机策略
            num_simulations: MCTS模拟次数
            c_puct: 探索系数
            temperature: 温度参数，None则根据deterministic自动设置
            deterministic: 是否使用确定性策略
            device: 运行设备
            model_type: 模型类型 (small/medium/large/advanced/hybrid)，None则自动检测
            num_parallel: 并行搜索线程数，>1时启用GPU批量加速
            use_rule_based: 是否使用规则辅助 (get_urgent_moves)
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.temperature = temperature
        self.deterministic = deterministic
        self.use_rule_based = use_rule_based

        # 加载模型
        if model_path and os.path.exists(model_path):
            print(f"加载模型: {model_path}")
            # 强制先映射到 CPU，这样可以剥离掉 NPU 的特定属性
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)

            # 自动检测或使用指定的模型类型
            if model_type is None:
                model_type = detect_model_type(state_dict)
            print(f"模型类型: {model_type}")

            self.network = create_network_by_type(model_type)
            self.network.load_state_dict(state_dict)
            self.network.to(device)
            self.network.eval()
        else:
            print("使用随机初始化模型")
            self.network = PolicyValueNet()
            self.network.to(device)
            self.network.eval()

        self.num_simulations = num_simulations
        self.c_puct = c_puct

        if num_parallel > 1:
            print(f"启用GPU加速MCTS (并行数: {num_parallel})")
            self.mcts = MCTSParallel(
                self.network,
                num_simulations=num_simulations,
                c_puct=c_puct,
                num_parallel=num_parallel,
                use_rule_based=use_rule_based,
            )
        else:
            self.mcts = MCTS(
                self.network,
                num_simulations=num_simulations,
                c_puct=c_puct,
                use_rule_based=use_rule_based,
            )

    def get_action(self, env: Connect6Env, deterministic: bool = None) -> int:
        """
        获取AI动作

        Args:
            env: 当前游戏状态
            deterministic: 是否确定性选择，None则使用初始化时的设置

        Returns:
            action: 选择的动作
        """
        if deterministic is None:
            deterministic = self.deterministic

        if self.temperature is not None:
            self.mcts.temperature = self.temperature
        else:
            self.mcts.temperature = 0.1 if deterministic else 1.0

        policy = self.mcts.search(env, add_noise=not deterministic)

        if deterministic:
            return np.argmax(policy)
        else:
            return np.random.choice(len(policy), p=policy)

    def get_action_with_policy(self, env: Connect6Env) -> tuple:
        """
        获取AI动作及策略分布

        Returns:
            (action, policy, value)
        """
        policy = self.mcts.search(env, add_noise=False)
        action = np.argmax(policy)

        # 获取价值估计
        state = env.get_state()
        _, value = self.network.predict(state)

        return action, policy, value


class HumanPlayer:
    """人类玩家"""

    def get_action(self, env: Connect6Env, deterministic: bool = True) -> int:
        """获取人类输入的动作"""
        while True:
            try:
                move = input("请输入落子位置 (格式: 行,列 或 字母+数字如A1): ").strip()

                if ',' in move:
                    # 格式: 行,列
                    parts = move.split(',')
                    x, y = int(parts[0]), int(parts[1])
                elif move[0].isalpha():
                    # 格式: 字母+数字 (如 A1, J10)
                    col = move[0].upper()
                    row = int(move[1:]) - 1
                    if col >= 'A' and col <= 'S':
                        y = ord(col) - ord('A')
                    else:
                        print("列必须在A-S之间")
                        continue
                    x = row
                else:
                    # 格式: 行 列
                    parts = move.split()
                    x, y = int(parts[0]), int(parts[1])

                action = env.coord_to_action(x, y)
                if env.is_valid_move(x, y):
                    return action
                else:
                    print("该位置已有棋子或超出范围，请重新输入")
            except (ValueError, IndexError):
                print("输入格式错误，请使用 '行,列' 或 'A1' 格式")


def play_game(
    player1,
    player2,
    verbose: bool = True,
    player1_name: str = "玩家1",
    player2_name: str = "玩家2",
):
    """
    进行一局对弈

    Args:
        player1: 黑方玩家
        player2: 白方玩家
        verbose: 是否显示详细信息
        player1_name: 黑方名称
        player2_name: 白方名称

    Returns:
        winner: 获胜方 (Player.BLACK, Player.WHITE, 或 None表示平局)
    """
    env = Connect6Env()
    players = {Player.BLACK: (player1, player1_name), Player.WHITE: (player2, player2_name)}

    if verbose:
        print("\n游戏开始!")
        print(f"黑方: {player1_name} (X)")
        print(f"白方: {player2_name} (O)")
        print(env.render())

    while not env.game_over:
        player, name = players[env.current_player]
        remaining = env.get_remaining_moves()

        if verbose:
            color = "黑方" if env.current_player == Player.BLACK else "白方"
            print(f"\n{color} ({name}) 落子 (剩余 {remaining} 子)")

        action = player.get_action(env)
        x, y = env.action_to_coord(action)

        if verbose:
            print(f"选择: ({x}, {y})")

        env.step(action)

        if verbose:
            print(env.render())

    # 显示结果
    if verbose:
        print("\n" + "=" * 40)
        if env.winner is None:
            print("游戏结束: 平局!")
        elif env.winner == Player.BLACK:
            print(f"游戏结束: 黑方 ({player1_name}) 获胜!")
        else:
            print(f"游戏结束: 白方 ({player2_name}) 获胜!")
        print("=" * 40)

    return env.winner


def main():
    parser = argparse.ArgumentParser(description='六子棋对弈')

    parser.add_argument('--mode', type=str, default='human-ai',
                        choices=['human-ai', 'ai-human', 'ai-ai', 'human-human'],
                        help='对弈模式')
    parser.add_argument('--model', type=str, default='checkpoints/final_model.pt',
                        help='AI模型路径')
    parser.add_argument('--model2', type=str, default=None,
                        help='第二个AI模型路径（AI对战时使用）')
    parser.add_argument('--simulations', type=int, default=800,
                        help='MCTS模拟次数')
    parser.add_argument('--c-puct', type=float, default=1.5,
                        help='MCTS探索系数')
    parser.add_argument('--temperature', type=float, default=None,
                        help='温度参数 (默认: 确定性模式0.1, 随机模式1.0)')
    parser.add_argument('--stochastic', action='store_true',
                        help='使用随机策略 (默认: 确定性策略)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='MCTS并行搜索线程数 (GPU加速，建议8-16)')
    parser.add_argument('--no-rule-based', action='store_true',
                        help='禁用规则辅助 (get_urgent_moves)')
    parser.add_argument('--device', type=str, default='auto',
                        help='运行设备')
    parser.add_argument('--games', type=int, default=1,
                        help='对弈局数（AI对战时使用）')

    args = parser.parse_args()

    # 创建玩家
    if 'ai' in args.mode:
        ai_player = Connect6Player(
            model_path=args.model,
            num_simulations=args.simulations,
            c_puct=args.c_puct,
            temperature=args.temperature,
            deterministic=not args.stochastic,
            device=args.device,
            num_parallel=args.parallel,
            use_rule_based=not args.no_rule_based,
        )

    if args.mode == 'ai-ai' and args.model2:
        ai_player2 = Connect6Player(
            model_path=args.model2,
            num_simulations=args.simulations,
            c_puct=args.c_puct,
            temperature=args.temperature,
            deterministic=not args.stochastic,
            device=args.device,
            num_parallel=args.parallel,
            use_rule_based=not args.no_rule_based,
        )

    human_player = HumanPlayer()

    # 设置对弈
    if args.mode == 'human-ai':
        player1, player2 = human_player, ai_player
        name1, name2 = "人类", "AI"
    elif args.mode == 'ai-human':
        player1, player2 = ai_player, human_player
        name1, name2 = "AI", "人类"
    elif args.mode == 'ai-ai':
        player1 = ai_player
        player2 = ai_player2 if args.model2 else ai_player
        name1 = "AI-1"
        name2 = "AI-2" if args.model2 else "AI-1"
    else:  # human-human
        player1, player2 = human_player, human_player
        name1, name2 = "玩家1", "玩家2"

    # 进行对弈
    if args.mode == 'ai-ai' and args.games > 1:
        # 多局对战统计
        results = {Player.BLACK: 0, Player.WHITE: 0, None: 0}

        for i in range(args.games):
            print(f"\n===== 第 {i+1}/{args.games} 局 =====")
            winner = play_game(player1, player2, verbose=False,
                             player1_name=name1, player2_name=name2)
            results[winner] += 1

            if winner == Player.BLACK:
                print(f"黑方 ({name1}) 获胜")
            elif winner == Player.WHITE:
                print(f"白方 ({name2}) 获胜")
            else:
                print("平局")

        print(f"\n===== 统计结果 ({args.games} 局) =====")
        print(f"黑方 ({name1}) 胜: {results[Player.BLACK]}")
        print(f"白方 ({name2}) 胜: {results[Player.WHITE]}")
        print(f"平局: {results[None]}")
    else:
        play_game(player1, player2, player1_name=name1, player2_name=name2)


if __name__ == "__main__":
    main()
