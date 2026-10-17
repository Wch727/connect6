"""
蒙特卡洛树搜索 (MCTS) 实现
基于AlphaZero的MCTS变体，使用神经网络指导搜索
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.connect6_env import Connect6Env, Player





class MCTSNode:
    """MCTS节点"""

    def __init__(self, prior: float = 0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children: Dict[int, 'MCTSNode'] = {}
        self.children_priors: Dict[int, float] = {}  # 存储所有合法动作的先验概率
        self.virtual_loss = 0  # 虚拟损失

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children_priors) > 0

    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        best_score = -float('inf')
        best_action = -1

        # 考虑虚拟损失的总访问量
        total_visits = self.visit_count + self.virtual_loss
        sqrt_total = math.sqrt(total_visits)

        # 遍历所有合法动作（包括未实例化的）
        for action, prior in self.children_priors.items():
            child = self.children.get(action)

            if child:
                # 已实例化的子节点
                child_visits = child.visit_count + child.virtual_loss
                if child_visits > 0:
                    q_value = (child.value_sum - child.virtual_loss) / child_visits
                    q_value = max(-1.0, min(1.0, q_value))
                else:
                    q_value = 0.0
                ucb_score = q_value + c_puct * prior * sqrt_total / (1 + child_visits)
            else:
                # 未实例化的子节点 (N=0, Q=0)
                ucb_score = c_puct * prior * sqrt_total  # Q=0, N=0 -> score = c * P * sqrt(N_total)

            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action

        # 延迟实例化：选中时才创建节点
        if best_action not in self.children:
            self.children[best_action] = MCTSNode(prior=self.children_priors[best_action])

        return best_action, self.children[best_action]

    def expand(self, policy: np.ndarray, valid_moves: np.ndarray):
        masked_policy = policy * valid_moves
        policy_sum = masked_policy.sum()
        if policy_sum > 0:
            masked_policy /= policy_sum
        else:
            if valid_moves.sum() == 0:
                raise RuntimeError("No valid moves but game not over!")
            masked_policy = valid_moves / valid_moves.sum()

        # 仅存储先验概率，不创建子节点对象
        for action in range(len(policy)):
            if valid_moves[action] > 0:
                self.children_priors[action] = masked_policy[action]

    def update(self, value: float):
        self.visit_count += 1
        self.value_sum += value


class MCTS:
    def __init__(
        self,
        network,
        c_puct: float = 2.0,
        num_simulations: int = 800,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 1.5,
        use_rule_based: bool = True,
    ):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.use_rule_based = use_rule_based

    def search(self, env: Connect6Env, add_noise: bool = True) -> np.ndarray:
        # 规则辅助：检查是否有必胜或必防点
        if self.use_rule_based:
            winning_moves, defensive_moves = env.get_urgent_moves()
        else:
            winning_moves, defensive_moves = [], []

        # 1. 如果有必胜点，直接返回（概率1.0）
        if winning_moves:
            action_size = env.get_action_space_size()
            policy = np.zeros(action_size, dtype=np.float32)
            # 随机选择一个必胜点（如果有多个）
            best_action = np.random.choice(winning_moves)
            policy[best_action] = 1.0
            return policy

        # 2. 如果有必防点，限制搜索范围或大幅增加先验
        # 多个防守候选交由搜索选择。
        # 这里我们不直接返回，而是让MCTS在这些点中选择（通过增加先验）

        root = MCTSNode()
        root_player = env.current_player  # 记录根节点玩家
        state = env.get_state()
        policy, value = self.network.predict(state)

        valid_moves = env.get_valid_moves()

        # 如果有必防点，将其他点的概率置为0（或者极小）
        if defensive_moves:
            # 创建掩码
            urgent_mask = np.zeros_like(policy)
            for action in defensive_moves:
                urgent_mask[action] = 1.0

            # 仅保留必防点的概率
            policy = policy * urgent_mask
            # 归一化
            if policy.sum() > 0:
                policy /= policy.sum()
            else:
                # 候选先验总和为零时均匀分配。
                policy[urgent_mask > 0] = 1.0 / len(defensive_moves)

            # 将根节点合法动作限制在防守候选内。

            valid_moves = valid_moves * urgent_mask

        root.expand(policy, valid_moves)
        if add_noise:
            self._add_dirichlet_noise(root, valid_moves)
        for _ in range(self.num_simulations):
            node = root
            search_path = [(node, root_player)]  # 存储(节点, 该节点对应的玩家)
            # 使用 fast_clone 替代 clone
            env_copy = env.fast_clone()
            while node.is_expanded() and not env_copy.game_over:
                action, node = node.select_child(self.c_puct)
                player_before_move = env_copy.current_player  # 落子前的玩家
                env_copy.step(action)
                search_path.append((node, player_before_move))
            if env_copy.game_over:
                if env_copy.winner is None:
                    value = 0.0
                else:
                    # 获取最后一个落子的玩家
                    last_player = search_path[-1][1]
                    if env_copy.winner == last_player:
                        value = 1.0  # 最后落子的玩家赢了
                    else:
                        value = -1.0
                leaf_player = search_path[-1][1]  # 叶节点玩家是最后落子的玩家
            else:
                state = env_copy.get_state()
                policy, value = self.network.predict(state)
                valid_moves = env_copy.get_valid_moves()
                if valid_moves.sum() > 0:
                    node.expand(policy, valid_moves)
                leaf_player = env_copy.current_player  # 叶节点玩家是当前待落子的玩家
            self._backup(search_path, value, leaf_player)
        return self._get_policy(root, env.get_action_space_size())

    def _add_dirichlet_noise(self, root: MCTSNode, valid_moves: np.ndarray):
        actions = list(root.children_priors.keys())
        if len(actions) == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for i, action in enumerate(actions):
            # 更新先验概率
            root.children_priors[action] = (
                (1 - self.dirichlet_epsilon) * root.children_priors[action] +
                self.dirichlet_epsilon * noise[i]
            )
            # 如果节点已创建，同步更新
            if action in root.children:
                root.children[action].prior = root.children_priors[action]

    def _backup(self, search_path: List[Tuple[MCTSNode, int]], value: float, leaf_player: int):
        """
        价值回传 - 针对六子棋的正确实现

        六子棋中同一玩家可能连续落子两次，因此不能简单地每步翻转价值。
        需要根据每个节点对应的玩家来决定是否翻转。

        Args:
            search_path: [(节点, 该节点对应的玩家), ...]
            value: 叶节点的评估值（对leaf_player的评估）
            leaf_player: 叶节点对应的玩家
        """
        current_value = value
        current_player = leaf_player

        for node, node_player in reversed(search_path):
            # 如果当前节点的玩家与上一个节点不同，翻转价值
            if node_player != current_player:
                current_value = -current_value
                current_player = node_player
            node.update(current_value)

    def _get_policy(self, root: MCTSNode, action_size: int) -> np.ndarray:
        visits = np.zeros(action_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        if self.temperature == 0:
            best_action = np.argmax(visits)
            policy = np.zeros(action_size, dtype=np.float32)
            policy[best_action] = 1.0
        else:
            visits_temp = visits ** (1.0 / self.temperature)
            policy = visits_temp / visits_temp.sum()
        return policy

    def get_action(self, env: Connect6Env, deterministic: bool = False) -> int:
        policy = self.search(env, add_noise=not deterministic)
        if deterministic:
            return np.argmax(policy)
        else:
            return np.random.choice(len(policy), p=policy)


class MCTSParallel:
    def __init__(
        self,
        network,
        c_puct: float = 1.5,
        num_simulations: int = 800,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.25,
        num_parallel: int = 8,
        use_rule_based: bool = True,
    ):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.num_parallel = num_parallel
        self.use_rule_based = use_rule_based

    def search(self, env: Connect6Env, add_noise: bool = True,
               temperature: float = 1.0) -> np.ndarray:
        # 规则辅助：检查是否有必胜或必防点
        if self.use_rule_based:
            winning_moves, defensive_moves = env.get_urgent_moves()

            # 1. 如果有必胜点，直接返回（概率1.0）
            if winning_moves:
                action_size = env.get_action_space_size()
                policy = np.zeros(action_size, dtype=np.float32)
                # 随机选择一个必胜点（如果有多个）
                best_action = np.random.choice(winning_moves)
                policy[best_action] = 1.0
                return policy

        root = MCTSNode()
        root_player = env.current_player  # 记录根节点玩家
        state = env.get_state()
        policy, value = self.network.predict(state)

        valid_moves = env.get_valid_moves()

        # 2. 如果有必防点，限制搜索范围
        if self.use_rule_based and 'defensive_moves' in locals() and defensive_moves:
            urgent_mask = np.zeros_like(valid_moves)
            for action in defensive_moves:
                urgent_mask[action] = 1.0
            valid_moves = valid_moves * urgent_mask

        root.expand(policy, valid_moves)
        if add_noise:
            self._add_dirichlet_noise(root, valid_moves)

        simulations_done = 0
        while simulations_done < self.num_simulations:
            leaf_data = []
            # 并行搜索
            for _ in range(min(self.num_parallel, self.num_simulations - simulations_done)):
                node = root
                node.virtual_loss += 1  # 根节点添加虚拟损失
                search_path = [(node, root_player)]
                # 使用 fast_clone 替代 clone，大幅减少CPU开销
                env_copy = env.fast_clone()

                while node.is_expanded() and not env_copy.game_over:
                    action, node = node.select_child(self.c_puct)
                    node.virtual_loss += 1  # 路径节点添加虚拟损失

                    player_before_move = env_copy.current_player
                    env_copy.step(action)
                    search_path.append((node, player_before_move))

                if not env_copy.game_over:
                    leaf_data.append((search_path, env_copy, node))
                else:
                    # 游戏结束，直接回传
                    if env_copy.winner is None:
                        value = 0.0
                    else:
                        last_player = search_path[-1][1]
                        if env_copy.winner == last_player:
                            value = 1.0
                        else:
                            value = -1.0
                    leaf_player = search_path[-1][1]
                    self._backup(search_path, value, leaf_player)
                    simulations_done += 1

            # 批量预测
            if leaf_data:
                states = np.array([data[1].get_state() for data in leaf_data])
                policies, values = self.network.predict_batch(states)

                for i, (search_path, env_copy, node) in enumerate(leaf_data):
                    valid_moves = env_copy.get_valid_moves()
                    if valid_moves.sum() > 0:
                        node.expand(policies[i], valid_moves)
                    leaf_player = env_copy.current_player
                    self._backup(search_path, values[i], leaf_player)
                    simulations_done += 1

        return self._get_policy(root, env.get_action_space_size(), temperature)

    def _add_dirichlet_noise(self, root: MCTSNode, valid_moves: np.ndarray):
        actions = list(root.children_priors.keys())
        if len(actions) == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for i, action in enumerate(actions):
            # 更新先验概率
            root.children_priors[action] = (
                (1 - self.dirichlet_epsilon) * root.children_priors[action] +
                self.dirichlet_epsilon * noise[i]
            )
            # 如果节点已创建，同步更新
            if action in root.children:
                root.children[action].prior = root.children_priors[action]

    def _backup(self, search_path: List[Tuple[MCTSNode, int]], value: float, leaf_player: int):
        """
        价值回传 - 包含虚拟损失的移除
        """
        current_value = value
        current_player = leaf_player

        for node, node_player in reversed(search_path):
            # 移除虚拟损失
            if node.virtual_loss > 0:
                node.virtual_loss -= 1

            if node_player != current_player:
                current_value = -current_value
                current_player = node_player
            node.update(current_value)

    def _get_policy(self, root: MCTSNode, action_size: int, temperature: float) -> np.ndarray:
        visits = np.zeros(action_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        if temperature == 0:
            best_action = np.argmax(visits)
            policy = np.zeros(action_size, dtype=np.float32)
            policy[best_action] = 1.0
        else:
            visits_temp = visits ** (1.0 / temperature)
            total = visits_temp.sum()
            if total > 0:
                policy = visits_temp / total
            else:
                policy = np.ones(action_size, dtype=np.float32) / action_size
        return policy


def test_mcts():
    print("测试MCTS...")
    import torch
    from model.network import PolicyValueNetSmall
    net = PolicyValueNetSmall()
    mcts = MCTS(net, num_simulations=100)
    env = Connect6Env()
    print("初始棋盘:")
    print(env.render())
    for i in range(5):
        policy = mcts.search(env, add_noise=True)
        action = np.random.choice(len(policy), p=policy)
        x, y = env.action_to_coord(action)
        print(f"\n搜索结果: 选择 ({x}, {y})")
        print(f"Top 5 动作概率: {sorted(enumerate(policy), key=lambda x: -x[1])[:5]}")
        env.step(action)
        print(env.render())
        if env.game_over:
            break


if __name__ == "__main__":
    test_mcts()
