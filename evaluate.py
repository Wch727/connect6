#!/usr/bin/env python3
"""
模型评估脚本
用于评估训练好的模型强度
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.connect6_env import Connect6Env, Player
from model.network import (
    PolicyValueNet, PolicyValueNetSmall, PolicyValueNetLarge,
    PolicyValueNetAdvanced, PolicyValueNetHybrid
)
from mcts.mcts import MCTS


def detect_model_type(state_dict):
    """
    根据模型权重自动检测模型类型

    Returns:
        model_type: 'small', 'medium', 'large', 'advanced', 'hybrid'
    """
    keys = list(state_dict.keys())

    # 检测hybrid模型（有stem和stage1/stage2/stage3）
    if any('stem' in k for k in keys) and any('stage1' in k for k in keys):
        return 'hybrid'

    # 检测advanced模型（有transformer_blocks）
    if any('transformer_blocks' in k for k in keys):
        return 'advanced'

    # 检测标准模型（small/medium/large）
    if 'input_conv.conv.weight' in keys:
        channels = state_dict['input_conv.conv.weight'].shape[0]
        if channels <= 64:
            return 'small'
        elif channels <= 128:
            return 'medium'
        else:
            return 'large'

    return 'medium'  # 默认


class RandomPlayer:
    """随机玩家（基准）"""

    def get_action(self, env: Connect6Env, deterministic: bool = True) -> int:
        valid_moves = env.get_valid_moves()
        valid_actions = np.where(valid_moves > 0)[0]
        return np.random.choice(valid_actions)


class GreedyPlayer:
    """贪婪玩家 - 使用简单启发式"""

    def __init__(self):
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    def get_action(self, env: Connect6Env, deterministic: bool = True) -> int:
        valid_moves = env.get_valid_moves()
        scores = np.zeros(361)

        for action in range(361):
            if valid_moves[action] > 0:
                x, y = action // 19, action % 19
                scores[action] = self._evaluate_position(env, x, y)

        # 添加一点随机性
        if not deterministic:
            noise = np.random.random(361) * 0.1
            scores = scores + noise * valid_moves

        return np.argmax(scores)

    def _evaluate_position(self, env: Connect6Env, x: int, y: int) -> float:
        """评估位置价值"""
        score = 0.0
        player = env.current_player
        opponent = -player

        for dx, dy in self.directions:
            # 检查己方连子
            my_count, my_space = self._count_line(env, x, y, dx, dy, player)
            # 检查对方连子
            opp_count, opp_space = self._count_line(env, x, y, dx, dy, opponent)

            # 评分
            if my_count >= 5:
                score += 100000  # 能赢
            elif my_count >= 4 and my_space >= 6:
                score += 10000
            elif my_count >= 3 and my_space >= 6:
                score += 1000

            if opp_count >= 5:
                score += 50000  # 防守
            elif opp_count >= 4 and opp_space >= 6:
                score += 5000
            elif opp_count >= 3 and opp_space >= 6:
                score += 500

        # 中心位置bonus
        center_dist = abs(x - 9) + abs(y - 9)
        score += max(0, 10 - center_dist)

        return score

    def _count_line(self, env: Connect6Env, x: int, y: int, dx: int, dy: int, player: int):
        """统计一条线上的连子数和空间"""
        count = 0
        space = 1

        # 正方向
        nx, ny = x + dx, y + dy
        while 0 <= nx < 19 and 0 <= ny < 19 and space < 6:
            if env.board[nx, ny] == player:
                count += 1
            elif env.board[nx, ny] == 0:
                pass
            else:
                break
            space += 1
            nx, ny = nx + dx, ny + dy

        # 反方向
        nx, ny = x - dx, y - dy
        while 0 <= nx < 19 and 0 <= ny < 19 and space < 6:
            if env.board[nx, ny] == player:
                count += 1
            elif env.board[nx, ny] == 0:
                pass
            else:
                break
            space += 1
            nx, ny = nx - dx, ny - dy

        return count, space


class AIPlayer:
    """AI玩家"""

    def __init__(self, model_path: str, num_simulations: int = 400, device: str = 'auto',
                 use_rule_based: bool = True):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # 加载模型
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # 自动检测模型类型
        model_type = detect_model_type(state_dict)
        print(f"检测到模型类型: {model_type}")

        # 创建对应的网络
        if model_type == 'hybrid':
            self.network = PolicyValueNetHybrid()
        elif model_type == 'advanced':
            self.network = PolicyValueNetAdvanced()
        elif model_type == 'small':
            self.network = PolicyValueNetSmall()
        elif model_type == 'large':
            self.network = PolicyValueNetLarge()
        else:  # medium
            self.network = PolicyValueNet(num_channels=128, num_res_blocks=10)

        self.network.load_state_dict(state_dict)
        self.network.to(device)
        self.network.eval()

        self.mcts = MCTS(self.network, num_simulations=num_simulations,
                         use_rule_based=use_rule_based)

    def get_action(self, env: Connect6Env, deterministic: bool = True) -> int:
        self.mcts.temperature = 0.1 if deterministic else 1.0
        policy = self.mcts.search(env, add_noise=not deterministic)
        return np.argmax(policy) if deterministic else np.random.choice(len(policy), p=policy)


def play_match(player1, player2, num_games: int = 100, verbose: bool = False):
    """进行多局对战"""
    results = {Player.BLACK: 0, Player.WHITE: 0, None: 0}

    for game_idx in tqdm(range(num_games), desc="对战进行中"):
        env = Connect6Env()
        players = {Player.BLACK: player1, Player.WHITE: player2}

        while not env.game_over:
            player = players[env.current_player]
            action = player.get_action(env)
            env.step(action)

        results[env.winner] += 1

        if verbose:
            winner = "黑方" if env.winner == Player.BLACK else ("白方" if env.winner == Player.WHITE else "平局")
            print(f"第{game_idx+1}局: {winner}")

    return results


def main():
    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--opponent', type=str, default='random',
                        choices=['random', 'greedy', 'self'],
                        help='对手类型')
    parser.add_argument('--games', type=int, default=100, help='对战局数')
    parser.add_argument('--simulations', type=int, default=400, help='MCTS模拟次数')
    parser.add_argument('--device', type=str, default='auto', help='设备')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    parser.add_argument('--opponent-model', type=str, help='另一模型的路径，优先于 --opponent')
    parser.add_argument('--no-rule-based', action='store_true', help='双方模型均关闭规则辅助')
    parser.add_argument('--seed', type=int, default=42, help='评测随机种子')
    parser.add_argument('--output', type=str, help='可选：保存设置、结果和落子序列的 JSON 文件')

    args = parser.parse_args()

    print(f"加载模型: {args.model}")
    if args.games <= 0 or args.simulations <= 0:
        parser.error('--games 和 --simulations 必须为正数')
    use_rules = not args.no_rule_based
    ai_player = AIPlayer(args.model, args.simulations, args.device, use_rules)

    # 创建对手
    if args.opponent_model:
        opponent = AIPlayer(args.opponent_model, args.simulations, args.device, use_rules)
        opponent_name = os.path.basename(args.opponent_model)
    elif args.opponent == 'random':
        opponent = RandomPlayer()
        opponent_name = "随机玩家"
    elif args.opponent == 'greedy':
        opponent = GreedyPlayer()
        opponent_name = "贪婪玩家"
    else:
        opponent = AIPlayer(args.model, args.simulations, args.device, use_rules)
        opponent_name = "自我对战"

    print(f"\n对手: {opponent_name}")
    print(f"对战局数: {args.games}")
    print(f"MCTS模拟次数: {args.simulations}")
    print("-" * 40)

    from evaluation import play_match_series
    results = play_match_series(ai_player, opponent, args.games, args.seed, args.verbose)
    ai_wins, opponent_wins, draws = results['wins'], results['losses'], results['draws']

    print("\n" + "=" * 40)
    print("评估结果汇总")
    print("=" * 40)
    print(f"AI 胜: {ai_wins} ({ai_wins/args.games*100:.1f}%)")
    print(f"{opponent_name} 胜: {opponent_wins} ({opponent_wins/args.games*100:.1f}%)")
    print(f"平局: {draws} ({draws/args.games*100:.1f}%)")
    print(f"得分率（平局计半分）: {results['score_rate']:.1%}")
    print(f"先手/后手局数: {results['games_as_black']}/{results['games_as_white']}")
    if args.output:
        import json
        from pathlib import Path
        report = {
            'model': os.path.basename(args.model), 'opponent': opponent_name,
            'simulations': args.simulations, 'rule_based': use_rules,
            'seed': args.seed, 'results': results,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"评测记录已保存: {output}")


if __name__ == "__main__":
    main()
