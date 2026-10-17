#!/usr/bin/env python3
"""
智奕连城 六子棋 AI 对弈争锋 - 比赛专用客户端 (V12.0 绝杀融合版)
---------------------------------------------------------
核心特性：
1. 界面：保留了原GUI的所有功能（计时器、坐标系、GB2312棋谱保存）。
2. 内核：替换为 Rule-Based + Numba 加速引擎，无需 .pt 模型，无需 GPU。
3. 算法：包含 Scan-Win 绝杀检测、强制防守、启发式评估。
"""

import pygame
import sys
import os
import time
import threading
import traceback
import argparse
import numpy as np
import datetime
import itertools

# ==========================================
# 0. 核心配置与加速引擎 (Numba)
# ==========================================
HAS_NUMBA = False
try:
    from numba import jit
    HAS_NUMBA = True
    print("✅ V12.0 绝杀执行引擎已启动 (Numba Accelerated)")
except ImportError:
    print("⚠️ 未检测到 Numba，AI 反应可能会慢。建议 pip install numba")
    def jit(nopython=True, cache=True):
        def decorator(func): return func
        return decorator

# --- 比赛规则配置 ---
WINDOW_WIDTH = 1300
WINDOW_HEIGHT = 900
GRID_SIZE = 42       # 格子大小
BOARD_SIZE = 19      # 19路棋盘
BOARD_PIXEL_SIZE = GRID_SIZE * (BOARD_SIZE - 1)
START_X = 80         # 棋盘左偏移
START_Y = 60         # 棋盘上偏移
TIME_LIMIT = 600     # 比赛限时: 600秒 (10分钟)

# --- 评分权重 ---
SCORE_WIN = 100_000_000
SCORE_LIVE_5 = 20_000_000
SCORE_DEAD_5 = 5_000_000
SCORE_LIVE_4 = 1_000_000
SCORE_DEAD_4 = 500_000
SCORE_LIVE_3 = 100_000
SCORE_DOUBLE = 500_000

# --- 视觉配色 ---
COLORS = {
    'bg': (236, 240, 241),           'board': (220, 179, 92),
    'grid': (0, 0, 0),               'black': (10, 10, 10),
    'white': (245, 245, 245),        'highlight': (231, 76, 60),
    'text': (44, 62, 80),            'timer_normal': (46, 204, 113),
    'timer_low': (231, 76, 60),      'btn_blue': (52, 152, 219),
    'btn_red': (231, 76, 60),        'btn_green': (46, 204, 113),
    'btn_yellow': (241, 196, 15)
}

# ---------------- Numba 加速算法区 ----------------

@jit(nopython=True, cache=True)
def scan_for_win_spots(board, player):
    """扫描是否有能够直接获胜的点位（凑够6子）"""
    win_spots_r = []
    win_spots_c = []
    size = 19
    dr = [0, 1, 1, 1]
    dc = [1, 0, 1, -1]

    for r in range(size):
        for c in range(size):
            if board[r, c] == player:
                for i in range(4):
                    stone_count = 0
                    has_opp = False
                    # 检查以 (r,c) 为起点的 6 格窗口
                    # 只有当窗口完全在棋盘内才检查
                    if not (0 <= r + 5 * dr[i] < size and 0 <= c + 5 * dc[i] < size):
                        continue

                    for k in range(6):
                        nr, nc = r + k*dr[i], c + k*dc[i]
                        val = board[nr, nc]
                        if val == player: stone_count += 1
                        elif val != 0:
                            has_opp = True; break

                    if not has_opp and stone_count >= 4:
                        for k in range(6):
                            nr, nc = r + k*dr[i], c + k*dc[i]
                            if board[nr, nc] == 0:
                                win_spots_r.append(nr)
                                win_spots_c.append(nc)
    return win_spots_r, win_spots_c

@jit(nopython=True, cache=True)
def fast_evaluate(board, player):
    """V12 快速评估函数"""
    total_score = 0
    size = 19
    dr = [0, 1, 1, 1]
    dc = [1, 0, 1, -1]
    my_threats = 0

    for r in range(size):
        for c in range(size):
            if board[r, c] == 0: continue
            current_color = board[r, c]
            is_me = (current_color == player)
            factor = 1.0 if is_me else 1.2 # 略微高估对手，防守优先

            for i in range(4):
                # 避免重复计算线段
                if 0 <= r - dr[i] < size and 0 <= c - dc[i] < size and board[r - dr[i], c - dc[i]] == current_color:
                    continue

                count = 0
                tr, tc = r, c
                while 0 <= tr < size and 0 <= tc < size and board[tr, tc] == current_color:
                    count += 1
                    tr += dr[i]
                    tc += dc[i]

                open_ends = 0
                if 0 <= r - dr[i] < size and 0 <= c - dc[i] < size and board[r - dr[i], c - dc[i]] == 0: open_ends += 1
                if 0 <= tr < size and 0 <= tc < size and board[tr, tc] == 0: open_ends += 1

                points = 0
                threat = False
                if count >= 6: points = SCORE_WIN
                elif count == 5:
                    points = SCORE_LIVE_5 if open_ends > 0 else SCORE_DEAD_5
                    threat = True
                elif count == 4:
                    points = SCORE_LIVE_4 if open_ends == 2 else SCORE_DEAD_4
                    threat = True
                elif count == 3:
                    points = SCORE_LIVE_3 if open_ends == 2 else 5000
                    if open_ends == 2: threat = True
                elif count == 2 and open_ends == 2: points = 500

                if is_me:
                    total_score += points
                    if threat: my_threats += 1
                else:
                    total_score -= points * factor

    if my_threats >= 2: total_score += SCORE_DOUBLE * (my_threats - 1)
    return total_score

@jit(nopython=True, cache=True)
def get_nearby_empty_points(board, radius=2):
    """获取有子区域附近的空点"""
    points = []
    size = 19
    # 用简单的mask标记
    mask = np.zeros((size, size), dtype=np.int8)
    has_stones = False

    for r in range(size):
        for c in range(size):
            if board[r, c] != 0:
                has_stones = True
                r_min, r_max = max(0, r-radius), min(size, r+radius+1)
                c_min, c_max = max(0, c-radius), min(size, c+radius+1)
                # 手动循环填充mask
                for i in range(r_min, r_max):
                    for j in range(c_min, c_max):
                        if board[i, j] == 0: mask[i, j] = 1

    if not has_stones: return [(9, 9)] # 天元

    # 收集点
    for r in range(size):
        for c in range(size):
            if mask[r, c] == 1:
                points.append((r, c))
    return points


# ---------------- UI 组件 ----------------

class CompetitionButton:
    def __init__(self, x, y, w, h, text, func, color_theme='btn_blue'):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.func = func
        self.base_color = COLORS.get(color_theme, COLORS['btn_blue'])
        self.hover = False

    def draw(self, screen, font):
        color = tuple(min(c + 20, 255) for c in self.base_color) if self.hover else self.base_color
        pygame.draw.rect(screen, (180, 180, 180), (self.rect.x, self.rect.y+4, self.rect.w, self.rect.h), border_radius=6)
        pygame.draw.rect(screen, color, self.rect, border_radius=6)
        txt_surf = font.render(self.text, True, (255, 255, 255))
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        screen.blit(txt_surf, txt_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hover: self.func()


class Connect6GUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("智奕连城 六子棋大赛专用终端 (V12.0 Rule-Based)")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # 字体加载
        font_names = ["SimHei", "Microsoft YaHei", "Arial"]
        self.font_title = pygame.font.SysFont(font_names, 36, bold=True)
        self.font_main = pygame.font.SysFont(font_names, 24)
        self.font_small = pygame.font.SysFont(font_names, 16)

        # 游戏状态
        # 注意：这里我们不再依赖 Connect6Env，直接用 numpy 数组管理，更直接
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1 # 1: 黑, -1: 白 (跟Rule代码对齐)
        self.turn_count = 0
        self.last_moves = [] # 记录步数 [(r, c), ...]
        self.history_colors = [] # 记录每步是谁下的

        self.ai_player_color = -1 # 默认AI执白
        self.mode = "human_vs_ai"

        self.game_active = False
        self.ai_thinking = False
        self.winner = None
        self.pending_moves = [] # 当前回合待确认的落子

        # 计时器
        self.time_black = TIME_LIMIT
        self.time_white = TIME_LIMIT
        self.last_time_update = 0
        self.team_black = "Player"
        self.team_white = "AI"

        self.ai_result_moves = None # AI 算完的结果

        # UI组件
        self.buttons = []
        self.init_ui()

    def init_ui(self):
        panel_x = START_X + BOARD_PIXEL_SIZE + 50
        panel_y = 520
        w, h, gap = 220, 50, 20

        self.btn_mode = CompetitionButton(panel_x, panel_y, w, h, "模式: 人执黑 vs AI", self.toggle_mode, 'btn_blue')
        self.buttons.append(self.btn_mode)
        self.buttons.append(CompetitionButton(panel_x, panel_y + h + gap, w, h, "开始 / 重置比赛", self.reset_game, 'btn_red'))
        self.buttons.append(CompetitionButton(panel_x, panel_y + (h + gap)*2, w, h, "保存比赛棋谱", self.save_log, 'btn_green'))
        self.buttons.append(CompetitionButton(panel_x, panel_y + (h + gap)*3, w, h, "强制 AI 落子", self.force_ai, 'btn_yellow'))

    def toggle_mode(self):
        modes = ["human_vs_ai", "ai_vs_human", "human_vs_human", "ai_vs_ai"]
        self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
        label_map = {
            "human_vs_ai": "模式: 人执黑 vs AI", "ai_vs_human": "模式: AI执黑 vs 人",
            "human_vs_human": "模式: 双人对弈", "ai_vs_ai": "模式: AI 内战"
        }
        self.btn_mode.text = label_map[self.mode]
        self.reset_game()

    def reset_game(self):
        self.board.fill(0)
        self.current_player = 1 # 黑先
        self.turn_count = 0
        self.last_moves = []
        self.history_colors = []
        self.pending_moves = []

        self.game_active = True
        self.winner = None
        self.ai_thinking = False
        self.time_black = TIME_LIMIT
        self.time_white = TIME_LIMIT
        self.last_time_update = time.time()

        if self.mode == "human_vs_ai": self.team_black, self.team_white = "Human", "AI_Bot"
        elif self.mode == "ai_vs_human": self.team_black, self.team_white = "AI_Bot", "Human"
        elif self.mode == "ai_vs_ai": self.team_black, self.team_white = "AI_B", "AI_W"
        else: self.team_black, self.team_white = "Player_A", "Player_B"
        print("比赛已重置。")

    def update_timer(self):
        if not self.game_active or self.winner is not None: return
        now = time.time()
        dt = now - self.last_time_update
        self.last_time_update = now

        if self.current_player == 1: self.time_black -= dt
        else: self.time_white -= dt

        if self.time_black <= 0:
            self.time_black = 0; self.winner = -1; self.game_active = False # 白胜
            print(">>> 黑方超时判负")
        elif self.time_white <= 0:
            self.time_white = 0; self.winner = 1; self.game_active = False # 黑胜
            print(">>> 白方超时判负")

    def get_needed_stones(self):
        # 规则：第一手黑1子，之后每人2子
        return 1 if self.turn_count == 0 else 2

    def check_win(self, player):
        # 检查是否获胜 (连6)
        size = 19
        dirs = [(0,1), (1,0), (1,1), (1,-1)]
        for r in range(size):
            for c in range(size):
                if self.board[r, c] == player:
                    for dr, dc in dirs:
                        cnt = 0
                        for k in range(6):
                            nr, nc = r + k*dr, c + k*dc
                            if 0 <= nr < size and 0 <= nc < size and self.board[nr, nc] == player: cnt += 1
                            else: break
                        if cnt >= 6: return True
        return False

    def execute_move(self, moves):
        """执行落子"""
        if self.game_active:
            for r, c in moves:
                self.board[r, c] = self.current_player
                self.last_moves.append((r, c))
                self.history_colors.append(self.current_player)

            # 检查胜负
            if self.check_win(self.current_player):
                self.winner = self.current_player
                self.game_active = False
            elif np.all(self.board != 0):
                self.winner = 0 # 平局
                self.game_active = False

            # 换手
            if self.game_active:
                self.current_player = -self.current_player
                self.turn_count += 1
                self.pending_moves = []

    # ================= 核心：Rule-Based AI 逻辑 (直接嵌入) =================
    def ai_think_thread(self):
        try:
            needed = self.get_needed_stones()
            player = self.current_player

            # 1. 开局
            if needed == 1:
                cands = get_nearby_empty_points(self.board, 2)
                best, best_val = [(9,9)], -float('inf')
                for r, c in cands:
                    self.board[r, c] = player
                    val = fast_evaluate(self.board, player) - (abs(r-9)+abs(c-9))*15
                    self.board[r, c] = 0
                    if val > best_val: best_val, best = val, [(r, c)]
                self.ai_result_moves = best
                return

            # 2. 绝杀检测
            my_win_r, my_win_c = scan_for_win_spots(self.board, player)
            my_wins = list(set(zip(my_win_r, my_win_c)))
            if len(my_wins) >= 2:
                self.ai_result_moves = list(my_wins)[:2]
                return
            elif len(my_wins) == 1:
                # 凑够5子，填1子
                w = my_wins[0]
                cands = get_nearby_empty_points(self.board, 2)
                if w in cands: cands.remove(w)
                self.ai_result_moves = [w, cands[0] if cands else (0,0)]
                return

            # 3. 绝防检测
            opp_win_r, opp_win_c = scan_for_win_spots(self.board, -player)
            opp_wins = list(set(zip(opp_win_r, opp_win_c)))
            forced = opp_wins

            # 4. 常规搜索
            cands = get_nearby_empty_points(self.board, 3)
            # 必须包含防守点
            unique = set(cands)
            for p in forced: unique.add(p)
            cands = list(unique)

            # 筛选
            top_singles = []
            for r, c in cands:
                self.board[r, c] = player
                atk = fast_evaluate(self.board, player)
                self.board[r, c] = -player
                dfe = fast_evaluate(self.board, -player)
                self.board[r, c] = 0
                bonus = 5e7 if (r, c) in forced else 0
                top_singles.append(((r, c), atk + dfe + bonus))

            top_singles.sort(key=lambda x: x[1], reverse=True)
            width = 12 if not forced else max(12, len(forced)+5)
            top_k = [x[0] for x in top_singles[:width]]

            # 组合
            pairs = list(itertools.combinations(top_k, 2))
            if not pairs and top_k: pairs = [top_k[:2]]

            best_pair_score = -float('inf')
            best_pair = pairs[0] if pairs else []

            for p1, p2 in pairs:
                # 剪枝：若有威胁，必须堵至少一个
                if forced:
                    blocked = 0
                    if p1 in forced: blocked += 1
                    if p2 in forced: blocked += 1
                    if blocked == 0: continue

                self.board[p1] = player
                self.board[p2] = player
                curr = fast_evaluate(self.board, player)

                # 漏防检测
                if forced:
                    sr, sc = scan_for_win_spots(self.board, -player)
                    if len(sr) > 0: curr = -float('inf')

                if curr > -float('inf'):
                    opp = fast_evaluate(self.board, -player)
                    curr -= opp * 1.5

                self.board[p1] = 0
                self.board[p2] = 0

                if curr > best_pair_score:
                    best_pair_score = curr
                    best_pair = [p1, p2]

            self.ai_result_moves = list(best_pair)

        except Exception:
            traceback.print_exc()
        finally:
            self.ai_thinking = False

    def check_ai_turn(self):
        if not self.game_active or self.winner is not None or self.ai_thinking: return

        # 检查是否轮到AI
        should_ai = False
        if self.mode == "ai_vs_ai": should_ai = True
        elif self.mode == "human_vs_ai" and self.current_player == -1: should_ai = True
        elif self.mode == "ai_vs_human" and self.current_player == 1: should_ai = True

        if should_ai:
            self.ai_thinking = True
            threading.Thread(target=self.ai_think_thread, daemon=True).start()

        # 检查是否AI算完了
        if not self.ai_thinking and self.ai_result_moves:
            self.execute_move(self.ai_result_moves)
            self.ai_result_moves = None

    def force_ai(self):
        if self.game_active and not self.ai_thinking:
            self.ai_thinking = True
            threading.Thread(target=self.ai_think_thread, daemon=True).start()

    # --- 坐标转换 & 存谱 ---
    def coord_to_pixel(self, r, c):
        return START_X + c * GRID_SIZE, START_Y + r * GRID_SIZE

    def pixel_to_coord(self, px, py):
        col = round((px - START_X) / GRID_SIZE)
        row = round((py - START_Y) / GRID_SIZE)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            cx, cy = self.coord_to_pixel(row, col)
            if (px - cx)**2 + (py - cy)**2 < (GRID_SIZE/2)**2: return row, col
        return None

    def save_log(self):
        if not self.last_moves: return
        res_str = "平局"
        if self.winner == 1: res_str = "先手胜"
        elif self.winner == -1: res_str = "后手胜"

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"records/C6-{self.team_black} vs {self.team_white}-{res_str}-{ts}.txt"
        os.makedirs("records", exist_ok=True)

        moves_list = []
        for (r, c), color_val in zip(self.last_moves, self.history_colors):
            color = "B" if color_val == 1 else "W"
            coord = f"{chr(ord('A') + c)},{19 - r}"
            moves_list.append(f"{color}({coord})")

        content = f"{{[C6][{self.team_black}][{self.team_white}][{res_str}][{datetime.datetime.now().strftime('%Y.%m.%d %H:%M')} Local][Beijing AI Contest];\n{';'.join(moves_list)}}}"

        try:
            with open(fname, "w", encoding="gb2312") as f: f.write(content)
            print(f"✅ 棋谱保存: {fname}")
        except:
            with open(fname, "w", encoding="utf-8") as f: f.write(content)
            print(f"✅ 棋谱保存(UTF-8): {fname}")

    # --- 绘制 ---
    def draw(self):
        self.screen.fill(COLORS['bg'])
        # 棋盘
        shadow = pygame.Rect(START_X-24, START_Y-24, BOARD_PIXEL_SIZE+48, BOARD_PIXEL_SIZE+48)
        pygame.draw.rect(self.screen, (200, 200, 200), shadow, border_radius=5)
        pygame.draw.rect(self.screen, COLORS['board'], (START_X-20, START_Y-20, BOARD_PIXEL_SIZE+40, BOARD_PIXEL_SIZE+40), border_radius=5)

        for i in range(BOARD_SIZE):
            y = START_Y + i * GRID_SIZE
            x = START_X + i * GRID_SIZE
            pygame.draw.line(self.screen, COLORS['grid'], (START_X, y), (START_X + BOARD_PIXEL_SIZE, y), 1)
            pygame.draw.line(self.screen, COLORS['grid'], (x, START_Y), (x, START_Y + BOARD_PIXEL_SIZE), 1)
            self.screen.blit(self.font_small.render(str(19 - i), True, COLORS['text']), (START_X - 30, y - 8))
            self.screen.blit(self.font_small.render(chr(ord('A') + i), True, COLORS['text']), (x - 5, START_Y + BOARD_PIXEL_SIZE + 10))

        for r in [3, 9, 15]:
            for c in [3, 9, 15]:
                pygame.draw.circle(self.screen, COLORS['grid'], self.coord_to_pixel(r, c), 4)

        # 棋子
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.board[r, c]
                if p != 0:
                    col = COLORS['black'] if p == 1 else COLORS['white']
                    cx, cy = self.coord_to_pixel(r, c)
                    pygame.draw.circle(self.screen, col, (cx, cy), GRID_SIZE//2 - 2)
                    if p == -1: pygame.draw.circle(self.screen, (150,150,150), (cx, cy), GRID_SIZE//2 - 2, 1)

        # 高亮最后2步
        if self.last_moves:
            for r, c in self.last_moves[-2:]:
                pygame.draw.circle(self.screen, COLORS['highlight'], self.coord_to_pixel(r, c), 6)

        # 待确认的子
        for r, c in self.pending_moves:
            cx, cy = self.coord_to_pixel(r, c)
            s = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
            col = (10,10,10,100) if self.current_player==1 else (240,240,240,100)
            pygame.draw.circle(s, col, (GRID_SIZE//2, GRID_SIZE//2), GRID_SIZE//2 - 2)
            self.screen.blit(s, (cx-GRID_SIZE//2, cy-GRID_SIZE//2))

        # 侧边栏
        px = START_X + BOARD_PIXEL_SIZE + 50
        pygame.draw.rect(self.screen, (30,30,30), (px, 60, 220, 90), border_radius=8)
        col_b = COLORS['timer_normal'] if self.time_black > 60 else COLORS['timer_low']
        self.screen.blit(self.font_title.render(f"{int(self.time_black)//60:02d}:{int(self.time_black)%60:02d}", True, col_b), (px+60, 80))
        self.screen.blit(self.font_small.render(f"黑方: {self.team_black}", True, (200,200,200)), (px+20, 120))

        pygame.draw.rect(self.screen, (220,220,220), (px, 170, 220, 90), border_radius=8)
        col_w = COLORS['timer_normal'] if self.time_white > 60 else COLORS['timer_low']
        self.screen.blit(self.font_title.render(f"{int(self.time_white)//60:02d}:{int(self.time_white)%60:02d}", True, col_w), (px+60, 190))
        self.screen.blit(self.font_small.render(f"白方: {self.team_white}", True, (50,50,50)), (px+20, 230))

        status = "AI 绝杀运算中..." if self.ai_thinking else ("比赛结束" if self.winner else ("比赛进行中" if self.game_active else "等待开始"))
        if self.winner: status = f"{'黑方' if self.winner == 1 else '白方'}获胜!"
        self.screen.blit(self.font_main.render(status, True, COLORS['text']), (px, 300))

        for btn in self.buttons: btn.draw(self.screen, self.font_main)

    def run(self):
        while True:
            self.update_timer()
            self.check_ai_turn()

            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                for btn in self.buttons: btn.handle_event(event)

                if event.type == pygame.MOUSEBUTTONDOWN and self.game_active and not self.winner and not self.ai_thinking:
                    can_click = (self.mode=="human_vs_human") or \
                                (self.mode=="human_vs_ai" and self.current_player==1) or \
                                (self.mode=="ai_vs_human" and self.current_player==-1)
                    if can_click:
                        coord = self.pixel_to_coord(*event.pos)
                        if coord:
                            r, c = coord
                            if self.board[r, c] == 0 and (r, c) not in self.pending_moves:
                                self.pending_moves.append((r, c))
                                if len(self.pending_moves) == self.get_needed_stones():
                                    self.execute_move(self.pending_moves)

                        # 右键取消
                        if event.button == 3 and self.pending_moves: self.pending_moves.pop()

            self.draw()
            pygame.display.flip()
            self.clock.tick(30)

if __name__ == "__main__":
    gui = Connect6GUI()
    gui.run()
