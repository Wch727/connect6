#!/usr/bin/env python3
"""
智奕连城 - 启发式宗师版 (Heuristic V3.1 - Fixed)
---------------------------------------------------------
修复日志:
1. [BugFix] 修复了 scan_win 未定义的错误。
2. [优化] 保留了 scan_threats 用于高级威胁分析，补回 scan_win 用于快速漏防检测。
"""

import pygame
import sys
import os
import time
import threading
import traceback
import numpy as np
import datetime
import itertools

# --- 核心配置 ---
WINDOW_WIDTH, WINDOW_HEIGHT = 1300, 900
GRID_SIZE = 42
BOARD_SIZE = 19
BOARD_PIXEL_SIZE = GRID_SIZE * (BOARD_SIZE - 1)
START_X, START_Y = 80, 60
TIME_LIMIT = 600

COLORS = {
    'bg': (236, 240, 241), 'board': (220, 179, 92), 'grid': (0, 0, 0),
    'black': (15, 15, 15), 'white': (245, 245, 245), 'highlight': (231, 76, 60),
    'text': (44, 62, 80), 'timer_normal': (46, 204, 113), 'timer_low': (231, 76, 60),
    'btn_blue': (52, 152, 219), 'btn_red': (231, 76, 60), 'panel_bg': (255, 255, 255)
}

# --- 评分权重 ---
SCORE_WIN       = 100_000_000
SCORE_LIVE_5    = 20_000_000
SCORE_DEAD_5    = 20_000_000
SCORE_LIVE_4    = 2_000_000
SCORE_DEAD_4    = 500_000
SCORE_LIVE_3    = 400_000
SCORE_JUMP_3    = 380_000
SCORE_DEAD_3    = 5_000
SCORE_LIVE_2    = 2_000

# --- Numba 加速 ---
try:
    from numba import jit
    print("✅ V3.1 宗师引擎已启动 (Numba Accelerated)")
except ImportError:
    print("⚠️ 未安装 numba，请 pip install numba")
    def jit(nopython=True, cache=True):
        def decorator(f): return f
        return decorator

# -------------------------------------------------------------------------
#  核心算法区
# -------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def scan_win(board, player):
    """
    【快速绝杀检测】
    仅检查是否能连成6子。用于漏防检查，速度极快。
    返回: (rows, cols) 列表
    """
    wins_r, wins_c = [], []
    size = 19
    dr = [0, 1, 1, 1]; dc = [1, 0, 1, -1]

    for r in range(size):
        for c in range(size):
            if board[r, c] == player:
                for i in range(4):
                    # 边界检查
                    if not (0 <= r + 5 * dr[i] < size and 0 <= c + 5 * dc[i] < size): continue

                    sc, ho = 0, False
                    for k in range(6):
                        v = board[r+k*dr[i], c+k*dc[i]]
                        if v == player: sc += 1
                        elif v != 0: ho = True; break

                    if not ho and sc >= 4: # 4子+2空=6，即为绝杀机会
                        for k in range(6):
                            nr, nc = r+k*dr[i], c+k*dc[i]
                            if board[nr, nc] == 0:
                                wins_r.append(nr)
                                wins_c.append(nc)
    return wins_r, wins_c

@jit(nopython=True, cache=True)
def get_line_score(count, open_ends, has_gap):
    if count >= 6: return SCORE_WIN
    if count == 5: return SCORE_LIVE_5
    if count == 4:
        if open_ends == 2: return SCORE_LIVE_4
        if open_ends == 1: return SCORE_DEAD_4
    if count == 3:
        if open_ends == 2: return SCORE_JUMP_3 if has_gap else SCORE_LIVE_3
        if open_ends == 1: return SCORE_DEAD_3
    if count == 2:
        if open_ends == 2: return SCORE_LIVE_2
    return count * 10

@jit(nopython=True, cache=True)
def evaluate_board(board, player):
    total_score = 0
    size = 19
    dr = [0, 1, 1, 1]; dc = [1, 0, 1, -1]
    defense_factor = 2.5

    for r in range(size):
        for c in range(size):
            if board[r, c] == 0: continue
            color = board[r, c]
            is_me = (color == player)

            for i in range(4):
                pr, pc = r - dr[i], c - dc[i]
                if 0 <= pr < size and 0 <= pc < size and board[pr, pc] == color: continue

                stone_cnt, gap_cnt = 0, 0
                first_gap = False

                for k in range(6):
                    nr, nc = r + k*dr[i], c + k*dc[i]
                    if not (0 <= nr < size and 0 <= nc < size): break
                    val = board[nr, nc]
                    if val == color:
                        stone_cnt += 1
                        if gap_cnt > 0: first_gap = True
                    elif val == 0: gap_cnt += 1
                    else: break

                open_ends = 0
                if 0 <= pr < size and 0 <= pc < size and board[pr, pc] == 0: open_ends += 1
                next_r = r + (stone_cnt + gap_cnt) * dr[i]
                next_c = c + (stone_cnt + gap_cnt) * dc[i]
                if 0 <= next_r < size and 0 <= next_c < size and board[next_r, next_c] == 0: open_ends += 1

                score = get_line_score(stone_cnt, open_ends, first_gap)
                if is_me: total_score += score
                else: total_score -= score * defense_factor
    return total_score

@jit(nopython=True, cache=True)
def scan_threats(board, player):
    """
    【高级危机扫描】
    返回: (绝杀点列表, 活3/跳3威胁点列表)
    """
    kill_moves = []
    urgent_moves = []
    size = 19
    dr = [0, 1, 1, 1]; dc = [1, 0, 1, -1]

    for r in range(size):
        for c in range(size):
            if board[r, c] == player:
                for i in range(4):
                    if not (0 <= r + 5 * dr[i] < size and 0 <= c + 5 * dc[i] < size): continue

                    stones, gaps, has_opp = 0, 0, False
                    for k in range(6):
                        nr, nc = r+k*dr[i], c+k*dc[i]
                        val = board[nr, nc]
                        if val == player: stones += 1
                        elif val == 0: gaps += 1
                        else: has_opp = True; break

                    if not has_opp:
                        if stones >= 4:
                            for k in range(6):
                                nr, nc = r+k*dr[i], c+k*dc[i]
                                if board[nr, nc] == 0: kill_moves.append((nr, nc))
                        elif stones == 3:
                            # 只有两头空的活3才算 Urgent
                            pr, pc = r - dr[i], c - dc[i]
                            nr6, nc6 = r + 6*dr[i], c + 6*dc[i]
                            open_head = (0 <= pr < size and 0 <= pc < size and board[pr, pc] == 0)
                            open_tail = (0 <= nr6 < size and 0 <= nc6 < size and board[nr6, nc6] == 0)
                            if open_head or open_tail:
                                for k in range(6):
                                    nr, nc = r+k*dr[i], c+k*dc[i]
                                    if board[nr, nc] == 0: urgent_moves.append((nr, nc))
    return kill_moves, urgent_moves

@jit(nopython=True, cache=True)
def get_candidates(board, radius=2):
    pts = []
    mask = np.zeros((19, 19), dtype=np.int8)
    has = False
    for r in range(19):
        for c in range(19):
            if board[r, c] != 0:
                has = True
                for i in range(max(0, r-radius), min(19, r+radius+1)):
                    for j in range(max(0, c-radius), min(19, c+radius+1)):
                        if board[i, j] == 0: mask[i, j] = 1
    if not has: return [(9, 9)]
    for r in range(19):
        for c in range(19):
            if mask[r, c] == 1: pts.append((r, c))
    return pts

# --- UI 类 ---
class CompetitionButton:
    def __init__(self, x, y, w, h, text, func, color_theme='btn_blue'):
        self.rect = pygame.Rect(x, y, w, h); self.text, self.func = text, func; self.base_color = COLORS.get(color_theme, COLORS['btn_blue']); self.hover = False
    def draw(self, screen, font):
        c = tuple(min(v+20, 255) for v in self.base_color) if self.hover else self.base_color
        pygame.draw.rect(screen, (180,180,180), (self.rect.x, self.rect.y+4, self.rect.w, self.rect.h), border_radius=8)
        pygame.draw.rect(screen, c, self.rect, border_radius=8)
        t = font.render(self.text, True, (255,255,255)); screen.blit(t, t.get_rect(center=self.rect.center))
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION: self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and self.hover: self.func()

class Connect6HeuristicGUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("智奕连城 - 启发式宗师版 (Heuristic V3.1)")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont(["SimHei", "Arial"], 36, bold=True)
        self.font_mid = pygame.font.SysFont(["SimHei", "Arial"], 24)
        self.font_small = pygame.font.SysFont(["SimHei", "Arial"], 16)

        self.board = np.zeros((19, 19), dtype=np.int8)
        self.curr_p, self.turn = 1, 0
        self.last, self.hist_c, self.pending = [], [], []

        self.mode = "human_vs_ai"
        self.active, self.winner = False, None
        self.ai_thinking = False
        self.ai_action_queue = None
        self.has_saved = False

        self.t_b, self.t_w, self.last_t = TIME_LIMIT, TIME_LIMIT, 0
        self.name_b, self.name_w = "Player", "Heuristic_V3"

        self.buttons = []
        self.init_ui()

    def init_ui(self):
        px, py, w, h, g = START_X + BOARD_PIXEL_SIZE + 50, 520, 220, 55, 25
        self.btn_mode = CompetitionButton(px, py, w, h, "模式: 人执黑 vs AI", self.toggle_mode, 'btn_blue')
        self.buttons.append(self.btn_mode)
        self.buttons.append(CompetitionButton(px, py+h+g, w, h, "开始 / 重置比赛", self.reset_game, 'btn_red'))

    def toggle_mode(self):
        modes = ["human_vs_ai", "ai_vs_human", "human_vs_human", "ai_vs_ai"]
        self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
        self.btn_mode.text = {"human_vs_ai": "模式: 人执黑 vs AI", "ai_vs_human": "模式: AI执黑 vs 人", "human_vs_human": "模式: 双人", "ai_vs_ai": "模式: AI 内战"}[self.mode]
        self.reset_game()

    def reset_game(self):
        self.board.fill(0)
        self.curr_p, self.turn, self.last, self.hist_c, self.pending = 1, 0, [], [], []
        self.active, self.winner = True, None
        self.ai_thinking, self.ai_action_queue, self.has_saved = False, None, False
        self.t_b, self.t_w, self.last_t = TIME_LIMIT, TIME_LIMIT, time.time()
        if self.mode == "human_vs_ai": self.name_b, self.name_w = "Human", "Heuristic_V3"
        elif self.mode == "ai_vs_human": self.name_b, self.name_w = "Heuristic_V3", "Human"
        elif self.mode == "ai_vs_ai": self.name_b, self.name_w = "Heuristic_B", "Heuristic_W"
        else: self.name_b, self.name_w = "P1", "P2"

    def update_timer(self):
        if not self.active or self.winner: return
        now = time.time(); dt = now - self.last_t; self.last_t = now
        if self.curr_p == 1: self.t_b -= dt
        else: self.t_w -= dt
        if self.t_b <= 0: self.finish_game(winner=-1)
        elif self.t_w <= 0: self.finish_game(winner=1)

    def finish_game(self, winner):
        if not self.active: return
        self.winner, self.active, self.ai_thinking = winner, False, False
        self.auto_save_log()

    def check_win(self, p):
        dr = [0, 1, 1, 1]; dc = [1, 0, 1, -1]
        for r in range(19):
            for c in range(19):
                if self.board[r, c] == p:
                    for i in range(4):
                        cnt = 0
                        for k in range(6):
                            if 0<=r+k*dr[i]<19 and 0<=c+k*dc[i]<19 and self.board[r+k*dr[i], c+k*dc[i]]==p: cnt+=1
                            else: break
                        if cnt >= 6: return True
        return False

    def execute(self, moves):
        for r, c in moves:
            self.board[r, c] = self.curr_p
            self.last.append((r, c)); self.hist_c.append(self.curr_p)
        if self.check_win(self.curr_p): self.finish_game(self.curr_p)
        elif np.all(self.board != 0): self.finish_game(0)
        if self.active: self.curr_p = -self.curr_p; self.turn += 1; self.pending = []

    # --- 宗师 AI 思考逻辑 ---
    def ai_worker(self):
        try:
            needed = 1 if self.turn == 0 else 2
            p = self.curr_p

            # 1. 开局：抢天元
            if needed == 1:
                if self.board[9,9] == 0: self.ai_action_queue = [(9,9)]; return
                cands = get_candidates(self.board, 2)
                best, bv = cands[0], -float('inf')
                for r, c in cands:
                    self.board[r, c] = p; v = evaluate_board(self.board, p)
                    self.board[r, c] = 0
                    if v > bv: bv, best = v, (r, c)
                self.ai_action_queue = [best]; return

            # 2. 绝杀检测 (My Win)
            my_kills, my_threats = scan_threats(self.board, p)
            my_kills = list(set(my_kills))
            if len(my_kills) >= 2:
                self.ai_action_queue = list(my_kills)[:2]; return
            elif len(my_kills) == 1:
                w = my_kills[0]
                cands = get_candidates(self.board, 2)
                if w in cands: cands.remove(w)
                self.ai_action_queue = [w, cands[0] if cands else (0,0)]; return

            # 3. 绝防检测 (Opponent Win)
            opp_kills, opp_threats = scan_threats(self.board, -p)
            opp_kills = list(set(opp_kills))

            forced_moves = set(opp_kills)
            for t in opp_threats: forced_moves.add(t)
            forced_moves = list(forced_moves)

            # 4. 贪婪搜索
            cands = get_candidates(self.board, 2)
            cand_set = set(cands)
            for fm in forced_moves: cand_set.add(fm)
            cands = list(cand_set)

            top_singles = []
            for r, c in cands:
                self.board[r, c] = p
                score = evaluate_board(self.board, p)
                self.board[r, c] = 0
                if (r, c) in forced_moves: score += 500_000_000
                top_singles.append(((r, c), score))

            top_singles.sort(key=lambda x: x[1], reverse=True)
            top_k = [x[0] for x in top_singles[:20]]

            pairs = list(itertools.combinations(top_k, 2))
            if not pairs and top_k: pairs = [top_k[:2]]

            best_pair = pairs[0] if pairs else []
            best_pair_score = -float('inf')

            for p1, p2 in pairs:
                if opp_kills:
                    covered = 0
                    if p1 in opp_kills: covered += 1
                    if p2 in opp_kills: covered += 1
                    if covered == 0: continue
                elif len(forced_moves) > 0:
                    covered = 0
                    if p1 in forced_moves: covered += 1
                    if p2 in forced_moves: covered += 1
                    # 如果有威胁（如活3），尽量去堵，但不是强制必须全堵(如果我能反杀)
                    # 不过为了稳健，还是建议堵
                    pass

                self.board[p1] = p
                self.board[p2] = p
                curr_score = evaluate_board(self.board, p)

                # 漏防检查：如果我下完这两步，对手还能赢吗？
                # 修复点：这里调用了 scan_win
                if opp_kills:
                    rem_r, rem_c = scan_win(self.board, -p)
                    if len(rem_r) > 0: curr_score = -float('inf')

                self.board[p1] = 0
                self.board[p2] = 0

                if curr_score > best_pair_score:
                    best_pair_score = curr_score
                    best_pair = [p1, p2]

            self.ai_action_queue = list(best_pair)

        except: traceback.print_exc()

    def check_ai(self):
        if not self.active or self.winner or self.ai_thinking: return
        should = (self.mode == "ai_vs_ai") or (self.mode == "human_vs_ai" and self.curr_p == -1) or (self.mode == "ai_vs_human" and self.curr_p == 1)
        if should:
            self.ai_thinking = True
            self.ai_action_queue = None
            threading.Thread(target=self.ai_worker, daemon=True).start()

    def process_ai(self):
        if self.ai_thinking and self.ai_action_queue:
            self.execute(self.ai_action_queue)
            self.ai_thinking = False
            self.ai_action_queue = None

    def auto_save_log(self):
        if self.has_saved or not self.last: return
        res = "先手胜" if self.winner == 1 else ("后手胜" if self.winner == -1 else "平局")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_log = datetime.datetime.now().strftime("%Y.%m.%d %H:%M")
        fn = f"records/C6-{self.name_b} vs {self.name_w}-{res}-{ts}.txt"
        os.makedirs("records", exist_ok=True)
        mvs = []
        for (r, c), p in zip(self.last, self.hist_c):
            col = "B" if p==1 else "W"
            mvs.append(f"{col}({chr(ord('A')+c)},{19-r})")
        ct = f"{{[C6][{self.name_b}][{self.name_w}][{res}][{ts_log} Local][HeuristicV3];\n{';'.join(mvs)}}}"
        try:
            with open(fn, "w", encoding="gb2312") as f: f.write(ct)
            print(f"✅ 自动保存: {fn}")
        except:
            with open(fn, "w", encoding="utf-8") as f: f.write(ct)
        self.has_saved = True

    def draw(self):
        self.screen.fill(COLORS['bg'])
        pygame.draw.rect(self.screen, (200,200,200), (START_X-24, START_Y-24, BOARD_PIXEL_SIZE+48, BOARD_PIXEL_SIZE+48), border_radius=5)
        pygame.draw.rect(self.screen, COLORS['board'], (START_X-20, START_Y-20, BOARD_PIXEL_SIZE+40, BOARD_PIXEL_SIZE+40), border_radius=5)
        for i in range(BOARD_SIZE):
            y, x = START_Y + i * GRID_SIZE, START_X + i * GRID_SIZE
            pygame.draw.line(self.screen, COLORS['grid'], (START_X, y), (START_X + BOARD_PIXEL_SIZE, y), 1)
            pygame.draw.line(self.screen, COLORS['grid'], (x, START_Y), (x, START_Y + BOARD_PIXEL_SIZE), 1)
            self.screen.blit(self.font_small.render(str(19-i), True, COLORS['text']), (START_X-30, y-8))
            self.screen.blit(self.font_small.render(chr(ord('A')+i), True, COLORS['text']), (x-5, START_Y+BOARD_PIXEL_SIZE+10))
        for r in [3,9,15]:
            for c in [3,9,15]: pygame.draw.circle(self.screen, COLORS['grid'], (START_X+c*GRID_SIZE, START_Y+r*GRID_SIZE), 4)

        for r in range(19):
            for c in range(19):
                p = self.board[r, c]
                if p != 0:
                    co = COLORS['black'] if p == 1 else COLORS['white']
                    cx, cy = START_X+c*GRID_SIZE, START_Y+r*GRID_SIZE
                    pygame.draw.circle(self.screen, co, (cx, cy), GRID_SIZE//2-2)
                    if p == -1: pygame.draw.circle(self.screen, (150,150,150), (cx, cy), GRID_SIZE//2-2, 1)
        if self.last:
            for r, c in self.last[-2:]: pygame.draw.circle(self.screen, COLORS['highlight'], (START_X+c*GRID_SIZE, START_Y+r*GRID_SIZE), 6)
        for r, c in self.pending:
            s = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
            co = (10,10,10,100) if self.curr_p==1 else (240,240,240,100)
            pygame.draw.circle(s, co, (GRID_SIZE//2, GRID_SIZE//2), GRID_SIZE//2-2)
            self.screen.blit(s, (START_X+c*GRID_SIZE-GRID_SIZE//2, START_Y+r*GRID_SIZE-GRID_SIZE//2))

        px = START_X + BOARD_PIXEL_SIZE + 50
        pygame.draw.rect(self.screen, COLORS['panel_bg'], (px, 60, 220, 240), border_radius=10)
        self.screen.blit(self.font_mid.render("黑方", True, COLORS['text']), (px+20, 70))
        self.screen.blit(self.font_big.render(f"{int(self.t_b)//60:02d}:{int(self.t_b)%60:02d}", True, COLORS['timer_normal']), (px+20, 100))
        self.screen.blit(self.font_mid.render("白方", True, COLORS['text']), (px+20, 180))
        self.screen.blit(self.font_big.render(f"{int(self.t_w)//60:02d}:{int(self.t_w)%60:02d}", True, COLORS['timer_normal']), (px+20, 210))

        st = "AI 宗师思考中..." if self.ai_thinking else ("比赛结束" if self.winner else "比赛进行中")
        if self.winner: st = f"{'黑' if self.winner==1 else '白'}方胜"
        self.screen.blit(self.font_mid.render(st, True, COLORS['text']), (px, 340))

        if self.has_saved: self.screen.blit(self.font_small.render("自动保存 √", True, COLORS['timer_normal']), (px, 380))
        for b in self.buttons: b.draw(self.screen, self.font_mid)

    def run(self):
        while True:
            self.update_timer(); self.check_ai(); self.process_ai()
            for e in pygame.event.get():
                if e.type == pygame.QUIT: pygame.quit(); sys.exit()
                for b in self.buttons: b.handle_event(e)
                if e.type == pygame.MOUSEBUTTONDOWN and self.active and not self.winner and not self.ai_thinking:
                    can = (self.mode=="human_vs_human") or (self.mode=="human_vs_ai" and self.curr_p==1) or (self.mode=="ai_vs_human" and self.curr_p==-1)
                    if can:
                        c, r = round((e.pos[0]-START_X)/GRID_SIZE), round((e.pos[1]-START_Y)/GRID_SIZE)
                        if 0<=r<19 and 0<=c<19 and self.board[r, c]==0 and (r, c) not in self.pending:
                            self.pending.append((r, c))
                            if len(self.pending) == (1 if self.turn==0 else 2): self.execute(self.pending)
                        if e.button==3 and self.pending: self.pending.pop()
            self.draw(); pygame.display.flip(); self.clock.tick(30)

if __name__ == "__main__":
    Connect6HeuristicGUI().run()
