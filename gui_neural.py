#!/usr/bin/env python3
"""
智奕连城 - 深度学习版 (Neural GUI - Final Stable)
---------------------------------------------------------
架构升级：
1. [稳定] 引入多线程队列通信，解决 AI 思考时 UI 卡死问题。
2. [智能] 自动搜索并加载最新的 checkpoint 模型。
3. [合规] 比赛结束自动保存棋谱，符合比赛坐标系。
"""

import pygame
import sys
import os
import time
import threading
import traceback
import argparse
import glob
from datetime import datetime

# --- 路径修正 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from game.connect6_env import Connect6Env, Player
    from play import Connect6Player
except ImportError as e:
    print(f"❌ 导入失败: {e}\n请确保在项目根目录下运行，且存在 game/ 和 play.py。")
    sys.exit(1)

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

# --- UI 类 ---
class CompetitionButton:
    def __init__(self, x, y, w, h, text, func, color_theme='btn_blue'):
        self.rect = pygame.Rect(x, y, w, h)
        self.text, self.func = text, func
        self.base_color = COLORS.get(color_theme, COLORS['btn_blue'])
        self.hover = False

    def draw(self, screen, font):
        c = tuple(min(v+20, 255) for v in self.base_color) if self.hover else self.base_color
        pygame.draw.rect(screen, (180,180,180), (self.rect.x, self.rect.y+4, self.rect.w, self.rect.h), border_radius=8)
        pygame.draw.rect(screen, c, self.rect, border_radius=8)
        txt = font.render(self.text, True, (255,255,255))
        screen.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and self.hover:
            self.func()

class Connect6NeuralGUI:
    def __init__(self, model_path=None, device='auto'):
        pygame.init()
        pygame.display.set_caption("智奕连城 - 深度学习版 (Neural Stable)")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # 字体
        self.font_big = pygame.font.SysFont(["SimHei", "Arial"], 36, bold=True)
        self.font_mid = pygame.font.SysFont(["SimHei", "Arial"], 24)
        self.font_small = pygame.font.SysFont(["SimHei", "Arial"], 16)

        # 游戏核心
        self.env = Connect6Env()
        self.ai_player = None
        self.init_ai(model_path, device)

        # 状态控制
        self.mode = "human_vs_ai"
        self.active, self.winner = False, None

        # 线程通信 (解决卡顿的关键)
        self.ai_thinking = False
        self.ai_action_queue = None
        self.has_saved = False

        # 计时与队名
        self.t_b, self.t_w, self.last_t = TIME_LIMIT, TIME_LIMIT, 0
        self.name_b, self.name_w = "Player", "Neural_AI"

        self.buttons = []
        self.init_ui()

    def find_model_file(self):
        print("🔍 自动扫描模型文件...")
        paths = [
            os.path.join(CURRENT_DIR, "checkpoints", "*.pt"),
            os.path.join(CURRENT_DIR, "*.pt")
        ]
        candidates = []
        for p in paths: candidates.extend(glob.glob(p))

        if not candidates: return None
        # 按修改时间排序，取最新的
        candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return candidates[0]

    def init_ai(self, path, device):
        print("="*50 + "\n【AI 初始化 (Neural)】")
        final_path = path if path and os.path.exists(path) else self.find_model_file()

        if not final_path:
            print("❌ 未找到 .pt 模型！将使用随机策略。")
            self.ai_player = Connect6Player(model_path=None)
            return

        print(f"🚀 加载模型: {os.path.basename(final_path)}")
        try:
            # 这里的 simulations 决定思考深度
            # CPU建议: 50-100; GPU建议: 400-800
            sims = 100 if device == 'cpu' else 400
            self.ai_player = Connect6Player(model_path=final_path, num_simulations=sims, device=device)
            print(f"✅ 模型加载成功！模拟次数: {sims}")
        except Exception:
            traceback.print_exc()
            print("⚠️ 加载失败，回退到随机策略。")
            self.ai_player = Connect6Player(model_path=None)
        print("="*50)

    def init_ui(self):
        px, py, w, h, g = START_X + BOARD_PIXEL_SIZE + 50, 520, 220, 55, 25
        self.btn_mode = CompetitionButton(px, py, w, h, "模式: 人执黑 vs AI", self.toggle_mode, 'btn_blue')
        self.buttons.append(self.btn_mode)
        self.buttons.append(CompetitionButton(px, py+h+g, w, h, "开始 / 重置比赛", self.reset_game, 'btn_red'))

    def toggle_mode(self):
        modes = ["human_vs_ai", "ai_vs_human", "human_vs_human", "ai_vs_ai"]
        self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
        self.btn_mode.text = {
            "human_vs_ai": "模式: 人执黑 vs AI", "ai_vs_human": "模式: AI执黑 vs 人",
            "human_vs_human": "模式: 双人", "ai_vs_ai": "模式: AI 内战"
        }[self.mode]
        self.reset_game()

    def reset_game(self):
        self.env.reset()
        self.active, self.winner = True, None
        self.ai_thinking, self.ai_action_queue, self.has_saved = False, None, False
        self.t_b, self.t_w, self.last_t = TIME_LIMIT, TIME_LIMIT, time.time()

        if self.mode == "human_vs_ai": self.name_b, self.name_w = "Human", "Neural_AI"
        elif self.mode == "ai_vs_human": self.name_b, self.name_w = "Neural_AI", "Human"
        elif self.mode == "ai_vs_ai": self.name_b, self.name_w = "Neural_B", "Neural_W"
        else: self.name_b, self.name_w = "P1", "P2"
        print(">>> 比赛已重置")

    def update_timer(self):
        if not self.active or self.winner: return
        now = time.time(); dt = now - self.last_t; self.last_t = now
        if self.env.current_player == Player.BLACK: self.t_b -= dt
        else: self.t_w -= dt
        if self.t_b <= 0: self.finish_game(winner=Player.WHITE, reason="超时")
        elif self.t_w <= 0: self.finish_game(winner=Player.BLACK, reason="超时")

    def finish_game(self, winner, reason=""):
        if not self.active: return
        self.winner, self.active, self.ai_thinking = winner, False, False
        self.auto_save_log()

    # --- AI 线程逻辑 (Anti-Freeze) ---
    def ai_worker(self):
        """后台思考线程"""
        try:
            # 获取动作，不操作 env
            action = self.ai_player.get_action(self.env)
            self.ai_action_queue = action
        except Exception:
            traceback.print_exc()
            self.ai_thinking = False

    def check_ai(self):
        """触发 AI"""
        if not self.active or self.winner or self.ai_thinking: return

        should = (self.mode == "ai_vs_ai") or \
                 (self.mode == "human_vs_ai" and self.env.current_player == Player.WHITE) or \
                 (self.mode == "ai_vs_human" and self.env.current_player == Player.BLACK)

        if should:
            self.ai_thinking = True
            self.ai_action_queue = None
            threading.Thread(target=self.ai_worker, daemon=True).start()

    def process_ai(self):
        """主线程执行落子"""
        if self.ai_thinking and self.ai_action_queue is not None:
            self.execute(self.ai_action_queue)
            self.ai_thinking = False
            self.ai_action_queue = None

    def execute(self, action):
        self.env.step(action)
        if self.env.game_over:
            self.finish_game(self.env.winner)

    # --- 存谱 ---
    def auto_save_log(self):
        if self.has_saved or not self.env.last_moves: return
        res = "先手胜" if self.winner == Player.BLACK else ("后手胜" if self.winner == Player.WHITE else "平局")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"records/C6-{self.name_b} vs {self.name_w}-{res}-{ts}.txt"
        os.makedirs("records", exist_ok=True)

        moves = []
        for i, (r, c) in enumerate(self.env.last_moves):
            color = "B" if i == 0 else ("W" if ((i+1)//2)%2==1 else "B")
            moves.append(f"{color}({chr(ord('A')+c)},{19-r})")

        ct = f"{{[C6][{self.name_b}][{self.name_w}][{res}][{datetime.now().strftime('%Y.%m.%d %H:%M')} Local][Neural];\n{';'.join(moves)}}}"
        try:
            with open(fn, "w", encoding="gb2312") as f: f.write(ct)
            print(f"✅ 自动保存: {fn}")
        except:
            with open(fn, "w", encoding="utf-8") as f: f.write(ct)
        self.has_saved = True

    # --- 绘图 ---
    def coord_to_pixel(self, r, c):
        return START_X + c * GRID_SIZE, START_Y + r * GRID_SIZE

    def pixel_to_coord(self, px, py):
        col = round((px - START_X) / GRID_SIZE)
        row = round((py - START_Y) / GRID_SIZE)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            cx, cy = self.coord_to_pixel(row, col)
            if (px - cx)**2 + (py - cy)**2 < (GRID_SIZE/2)**2: return row, col
        return None

    def draw(self):
        self.screen.fill(COLORS['bg'])
        # Board
        pygame.draw.rect(self.screen, (200,200,200), (START_X-24, START_Y-24, BOARD_PIXEL_SIZE+48, BOARD_PIXEL_SIZE+48), border_radius=5)
        pygame.draw.rect(self.screen, COLORS['board'], (START_X-20, START_Y-20, BOARD_PIXEL_SIZE+40, BOARD_PIXEL_SIZE+40), border_radius=5)
        for i in range(BOARD_SIZE):
            y, x = START_Y + i * GRID_SIZE, START_X + i * GRID_SIZE
            pygame.draw.line(self.screen, COLORS['grid'], (START_X, y), (START_X + BOARD_PIXEL_SIZE, y), 1)
            pygame.draw.line(self.screen, COLORS['grid'], (x, START_Y), (x, START_Y + BOARD_PIXEL_SIZE), 1)
            self.screen.blit(self.font_small.render(str(19-i), True, COLORS['text']), (START_X-30, y-8))
            self.screen.blit(self.font_small.render(chr(ord('A')+i), True, COLORS['text']), (x-5, START_Y+BOARD_PIXEL_SIZE+10))
        for r in [3,9,15]:
            for c in [3,9,15]: pygame.draw.circle(self.screen, COLORS['grid'], self.coord_to_pixel(r, c), 4)

        # Pieces
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.env.board[r, c]
                if p != Player.EMPTY:
                    color = COLORS['black'] if p == Player.BLACK else COLORS['white']
                    cx, cy = self.coord_to_pixel(r, c)
                    pygame.draw.circle(self.screen, color, (cx, cy), GRID_SIZE//2-2)
                    if p == Player.WHITE: pygame.draw.circle(self.screen, (150,150,150), (cx, cy), GRID_SIZE//2-2, 1)

        # Highlight
        if self.env.last_moves:
            for r, c in (self.env.last_moves[-2:] if len(self.env.last_moves) >= 2 else self.env.last_moves):
                pygame.draw.circle(self.screen, COLORS['highlight'], self.coord_to_pixel(r, c), 6)

        # Sidebar
        px = START_X + BOARD_PIXEL_SIZE + 50
        pygame.draw.rect(self.screen, COLORS['panel_bg'], (px, 60, 220, 240), border_radius=10)
        self.screen.blit(self.font_mid.render("黑方", True, COLORS['text']), (px+20, 70))
        self.screen.blit(self.font_big.render(f"{int(self.t_b)//60:02d}:{int(self.t_b)%60:02d}", True, COLORS['timer_normal']), (px+20, 100))
        self.screen.blit(self.font_mid.render("白方", True, COLORS['text']), (px+20, 180))
        self.screen.blit(self.font_big.render(f"{int(self.t_w)//60:02d}:{int(self.t_w)%60:02d}", True, COLORS['timer_normal']), (px+20, 210))

        st = "AI 深度计算中..." if self.ai_thinking else ("比赛结束" if self.winner else "比赛进行中")
        if self.winner: st = f"{'黑' if self.winner==Player.BLACK else '白'}方胜"
        self.screen.blit(self.font_mid.render(st, True, COLORS['text']), (px, 340))

        if self.has_saved: self.screen.blit(self.font_small.render("自动保存 √", True, COLORS['timer_normal']), (px, 380))
        for b in self.buttons: b.draw(self.screen, self.font_mid)

    def run(self):
        while True:
            self.update_timer()
            self.check_ai()
            self.process_ai()

            for e in pygame.event.get():
                if e.type == pygame.QUIT: pygame.quit(); sys.exit()
                for b in self.buttons: b.handle_event(e)

                if e.type == pygame.MOUSEBUTTONDOWN and self.active and not self.winner and not self.ai_thinking:
                    can = (self.mode == "human_vs_human") or \
                          (self.mode == "human_vs_ai" and self.env.current_player == Player.BLACK) or \
                          (self.mode == "ai_vs_human" and self.env.current_player == Player.WHITE)
                    if can:
                        coord = self.pixel_to_coord(*e.pos)
                        if coord and self.env.is_valid_move(*coord):
                            self.execute(self.env.coord_to_action(*coord))

            self.draw()
            pygame.display.flip()
            self.clock.tick(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    Connect6NeuralGUI(model_path=args.model).run()
