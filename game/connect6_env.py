"""
六子棋游戏环境
规则：
1. 19x19棋盘
2. 黑方先行，第一手只下一子
3. 之后双方轮流，每次各下两子
4. 先连成六子者获胜
"""

import numpy as np
from typing import Tuple, List, Optional
from enum import IntEnum


class Player(IntEnum):
    BLACK = 1
    WHITE = -1
    EMPTY = 0


class Connect6Env:
    """六子棋游戏环境"""

    BOARD_SIZE = 19
    WIN_LENGTH = 6

    # 八个方向：横、竖、两条对角线
    DIRECTIONS = [
        (0, 1),   # 横向
        (1, 0),   # 纵向
        (1, 1),   # 主对角线
        (1, -1),  # 副对角线
    ]

    def __init__(self):
        self.reset()

    def reset(self) -> np.ndarray:
        """重置游戏状态"""
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self.current_player = Player.BLACK
        self.move_count = 0  # 当前回合已下的子数
        self.turn = 0  # 总回合数
        self.game_over = False
        self.winner = None
        self.last_moves = []  # 记录最近的落子
        self.history = []  # 记录所有历史状态
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        获取当前状态表示（AlphaZero 标准格式）
        返回形状为 (17, 19, 19) 的特征平面:
        - 平面 0-7: 当前玩家最近8步的落子（当前棋盘为第0步）
        - 平面 8-15: 对手玩家最近8步的落子
        - 平面 16: 全1（当前玩家视角标识）
        """
        state = np.zeros((17, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)

        # 构建棋盘序列：[当前棋盘, 历史1, ..., 历史7]（最多8个）
        board_history = [self.board] + self.history[-7:]

        for t in range(8):
            if t < len(board_history):
                board_t = board_history[t]
                if self.current_player == Player.BLACK:
                    own_pieces = (board_t == Player.BLACK)
                    opp_pieces = (board_t == Player.WHITE)
                else:
                    own_pieces = (board_t == Player.WHITE)
                    opp_pieces = (board_t == Player.BLACK)
            else:
                own_pieces = opp_pieces = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=bool)

            state[t] = own_pieces.astype(np.float32)
            state[t + 8] = opp_pieces.astype(np.float32)

        state[16] = 1.0  # 当前玩家视角标识（恒为1）
        return state

    def get_valid_moves(self) -> np.ndarray:
        """获取所有合法落子位置（一维掩码）"""
        valid = (self.board == Player.EMPTY).astype(np.float32)
        return valid.flatten()

    def get_valid_moves_2d(self) -> np.ndarray:
        """获取所有合法落子位置（二维）"""
        return (self.board == Player.EMPTY).astype(np.float32)

    def is_valid_move(self, x: int, y: int) -> bool:
        """检查落子是否合法"""
        if x < 0 or x >= self.BOARD_SIZE or y < 0 or y >= self.BOARD_SIZE:
            return False
        return self.board[x, y] == Player.EMPTY

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步落子
        action: 0 到 360 的整数
        返回: (新状态, 奖励, 是否结束, 信息字典)
        """
        if self.game_over:
            return self.get_state(), 0, True, {"winner": self.winner}

        x, y = action // self.BOARD_SIZE, action % self.BOARD_SIZE

        if not self.is_valid_move(x, y):
            return self.get_state(), -1, True, {"error": "invalid_move"}

        self.history.append(self.board.copy())
        self.board[x, y] = self.current_player
        self.last_moves.append((x, y))
        self.move_count += 1

        if self._check_win(x, y):
            self.game_over = True
            self.winner = self.current_player
            reward = 1.0
            return self.get_state(), reward, True, {"winner": self.winner}

        if np.sum(self.board == Player.EMPTY) == 0:
            self.game_over = True
            self.winner = None
            return self.get_state(), 0, True, {"winner": None}

        moves_this_turn = self._get_moves_this_turn()
        if self.move_count >= moves_this_turn:
            self.current_player = Player(-self.current_player)
            self.move_count = 0
            self.turn += 1

        return self.get_state(), 0, False, {}

    def _get_moves_this_turn(self) -> int:
        """获取当前回合需要下的子数"""
        if self.turn == 0 and self.current_player == Player.BLACK:
            return 1
        return 2

    def get_remaining_moves(self) -> int:
        """获取当前回合剩余需要下的子数"""
        return self._get_moves_this_turn() - self.move_count

    def _check_win(self, x: int, y: int) -> bool:
        """检查最后落子位置是否形成六连"""
        player = self.board[x, y]
        for dx, dy in self.DIRECTIONS:
            count = 1
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.BOARD_SIZE and 0 <= ny < self.BOARD_SIZE:
                if self.board[nx, ny] == player:
                    count += 1
                    nx, ny = nx + dx, ny + dy
                else:
                    break
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.BOARD_SIZE and 0 <= ny < self.BOARD_SIZE:
                if self.board[nx, ny] == player:
                    count += 1
                    nx, ny = nx - dx, ny - dy
                else:
                    break
            if count >= self.WIN_LENGTH:
                return True
        return False

    def clone(self) -> 'Connect6Env':
        """创建游戏状态的深拷贝"""
        new_env = Connect6Env()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.move_count = self.move_count
        new_env.turn = self.turn
        new_env.game_over = self.game_over
        new_env.winner = self.winner
        new_env.last_moves = self.last_moves.copy()
        new_env.history = [h.copy() for h in self.history]
        return new_env

    def fast_clone(self) -> 'Connect6Env':
        """
        快速拷贝 - 仅用于MCTS搜索
        优化：绕过 __init__ 避免不必要的 reset 和 get_state 计算
        """
        # 使用 __new__ 绕过 __init__，避免触发 reset() 和 get_state()
        new_env = object.__new__(Connect6Env)

        # 手动复制必要的属性
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.move_count = self.move_count
        new_env.turn = self.turn
        new_env.game_over = self.game_over
        new_env.winner = self.winner

        # MCTS不需要完整的last_moves，只需要知道最后几步用于显示（可选）
        new_env.last_moves = self.last_moves[-2:] if self.last_moves else []

        # 关键优化：只复制最近7步历史，满足get_state需求即可
        if len(self.history) > 7:
            new_env.history = self.history[-7:]
        else:
            # 浅拷贝列表，里面的array不copy（因为历史状态是不可变的）
            new_env.history = list(self.history)

        return new_env

    def render(self) -> str:
        """渲染棋盘为字符串"""
        symbols = {Player.EMPTY: '.', Player.BLACK: 'X', Player.WHITE: 'O'}
        lines = []
        col_labels = '   ' + ' '.join([chr(ord('A') + i) if i < 26 else chr(ord('a') + i - 26)
                                        for i in range(self.BOARD_SIZE)])
        lines.append(col_labels)
        for i in range(self.BOARD_SIZE):
            row = f'{i+1:2d} '
            for j in range(self.BOARD_SIZE):
                if (i, j) in self.last_moves[-2:]:
                    row += f'[{symbols[self.board[i, j]]}]'[1]
                else:
                    row += symbols[self.board[i, j]]
                row += ' '
            lines.append(row)
        status = f"\n当前玩家: {'黑方' if self.current_player == Player.BLACK else '白方'}"
        status += f" | 本回合已下: {self.move_count}/{self._get_moves_this_turn()}"
        lines.append(status)
        return '\n'.join(lines)

    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        return self.BOARD_SIZE * self.BOARD_SIZE

    def get_canonical_board(self) -> np.ndarray:
        """获取标准化棋盘（当前玩家视角）"""
        if self.current_player == Player.BLACK:
            return self.board.copy()
        else:
            return -self.board.copy()

    def action_to_coord(self, action: int) -> Tuple[int, int]:
        """将动作索引转换为坐标"""
        return action // self.BOARD_SIZE, action % self.BOARD_SIZE

    def coord_to_action(self, x: int, y: int) -> int:
        """将坐标转换为动作索引"""
        return x * self.BOARD_SIZE + y

    def get_urgent_moves(self) -> Tuple[List[int], List[int]]:
        """
        获取紧急落子位置（规则辅助）

        Returns:
            winning_moves: 能直接获胜的落子位置列表
            defensive_moves: 必须防守的落子位置列表
        """
        winning_moves = set()
        defensive_moves_5 = set()
        defensive_moves_4 = set()

        # 获取当前玩家和对手
        me = self.current_player
        opp = Player(-me)
        remaining_moves = self.get_remaining_moves()

        # 扫描所有方向
        # 优化：只扫描有棋子的区域附近？为了安全起见，扫描全盘（19x19其实很快）
        # 使用滑动窗口检测

        # 1. 检查己方是否能赢 (5连缺1，或4连缺2且剩余2步)
        # 2. 检查对手是否要赢 (4连缺2 或 5连缺1)

        # 遍历所有点作为潜在落子点
        # 这种方法比全盘卷积慢，但逻辑更简单
        # 优化：只检查空位

        # 更高效的方法：遍历所有线
        # 横向
        for r in range(self.BOARD_SIZE):
            row = self.board[r, :]
            self._scan_line(row, me, opp, lambda c: self.coord_to_action(r, c), winning_moves, defensive_moves_5, defensive_moves_4, remaining_moves)

        # 纵向
        for c in range(self.BOARD_SIZE):
            col = self.board[:, c]
            self._scan_line(col, me, opp, lambda r: self.coord_to_action(r, c), winning_moves, defensive_moves_5, defensive_moves_4, remaining_moves)

        # 主对角线
        # diag(k) -> k=0是主对角线, k>0是上方, k<0是下方
        for k in range(-(self.BOARD_SIZE-6), self.BOARD_SIZE-5):
            diag = np.diagonal(self.board, k)
            # 需要计算坐标偏移
            if k >= 0:
                offset_r, offset_c = 0, k
            else:
                offset_r, offset_c = -k, 0
            self._scan_line(diag, me, opp, lambda i: self.coord_to_action(offset_r+i, offset_c+i), winning_moves, defensive_moves_5, defensive_moves_4, remaining_moves)

        # 副对角线
        # flipud后取对角线
        flipped_board = np.flipud(self.board)
        for k in range(-(self.BOARD_SIZE-6), self.BOARD_SIZE-5):
            diag = np.diagonal(flipped_board, k)
            if k >= 0:
                # flipped: (0, k) -> original: (18-0, k) = (18, k)
                # flipped: (i, k+i) -> original: (18-i, k+i)
                start_r, start_c = self.BOARD_SIZE - 1, k
            else:
                # flipped: (-k, 0) -> original: (18-(-k), 0) = (18+k, 0)
                start_r, start_c = self.BOARD_SIZE - 1 + k, 0

            self._scan_line(diag, me, opp, lambda i: self.coord_to_action(start_r-i, start_c+i), winning_moves, defensive_moves_5, defensive_moves_4, remaining_moves)

        # 优先返回针对5连的防守
        if defensive_moves_5:
            return list(winning_moves), list(defensive_moves_5)
        return list(winning_moves), list(defensive_moves_4)

    def _scan_line(self, line: np.ndarray, me: int, opp: int, idx_to_action, winning_moves: set, defensive_moves_5: set, defensive_moves_4: set, remaining_moves: int):
        """扫描单行寻找关键点"""
        if len(line) < 6:
            return

        # 窗口大小为6
        # 我们可以使用卷积或者简单的滑动窗口
        # 这里使用简单的滑动窗口

        # 预计算空位索引
        empty_indices = np.where(line == Player.EMPTY)[0]
        # empty_set = set(empty_indices) # 未使用

        # 优化：如果一行全是空，跳过
        if len(empty_indices) == len(line):
            return

        for i in range(len(line) - 5):
            window = line[i:i+6]

            # 统计
            me_count = np.sum(window == me)
            opp_count = np.sum(window == opp)
            empty_count = 6 - me_count - opp_count

            # 1. 己方赢棋机会
            # 情况A: 5个己方 + 1个空位 (只需1步)
            if me_count == 5 and empty_count == 1:
                # 找到那个空位
                for j in range(6):
                    if window[j] == Player.EMPTY:
                        action = idx_to_action(i+j)
                        x, y = self.action_to_coord(action)
                        if self.is_valid_move(x, y):
                            winning_moves.add(action)

            # 情况B: 4个己方 + 2个空位 (如果剩余步数>=2，则可以连下两子获胜)
            if me_count == 4 and empty_count == 2 and remaining_moves >= 2:
                for j in range(6):
                    if window[j] == Player.EMPTY:
                        action = idx_to_action(i+j)
                        x, y = self.action_to_coord(action)
                        if self.is_valid_move(x, y):
                            winning_moves.add(action)

            # 2. 对手威胁
            # 情况A: 对手5个 + 1个空位 (对手下回合必胜，必须堵)
            if opp_count == 5 and empty_count == 1:
                for j in range(6):
                    if window[j] == Player.EMPTY:
                        action = idx_to_action(i+j)
                        x, y = self.action_to_coord(action)
                        if self.is_valid_move(x, y):
                            defensive_moves_5.add(action)

            # 情况B: 对手四子、两个空位，收集防守候选。
            # 注意：只有当这2个空位都能被对手利用时才危险
            # 但为了安全，只要是4+2，这两个空位都是关键防守点
            if opp_count == 4 and empty_count == 2:
                for j in range(6):
                    if window[j] == Player.EMPTY:
                        action = idx_to_action(i+j)
                        x, y = self.action_to_coord(action)
                        if self.is_valid_move(x, y):
                            defensive_moves_4.add(action)


def test_env():
    env = Connect6Env()
    print("初始棋盘:")
    print(env.render())
    moves = [(9, 9), (8, 8), (8, 10), (10, 8), (10, 10)]
    for x, y in moves:
        action = env.coord_to_action(x, y)
        state, reward, done, info = env.step(action)
        print(f"\n落子: ({x}, {y})")
        print(env.render())
        if done:
            print(f"游戏结束! 奖励: {reward}, 信息: {info}")
            break


if __name__ == "__main__":
    test_env()
