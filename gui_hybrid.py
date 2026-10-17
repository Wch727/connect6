#!/usr/bin/env python3
"""
智奕连城 - 专业比赛客户端 (Pro Final Fixed)
---------------------------------------------------------
修复:
1. [关键] 修复棋谱保存失败的问题 (使用绝对路径 + 文件名清洗)。
2. [关键] 增加保存成功/失败的控制台提示。
3. [功能] 包含完整的开局库、混合对抗、日志复刻功能。
"""

import pygame
import sys
import os
import time
import threading
import traceback
import glob
import pickle
import numpy as np
import re
from datetime import datetime

# --- 路径兼容处理 ---
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# --- 复刻日志 Banner ---
print("Hello from the pygame community. https://www.pygame.org/contribute.html")
print("="*60)
print(f"📂 程序运行根目录: {BASE_DIR}")
print(f"💾 棋谱将保存在:   {os.path.join(BASE_DIR, 'records')}")
print("="*60)

try:
    from game.connect6_env import Connect6Env, Player
    from play import Connect6Player
    import torch
except ImportError as e:
    print(f"❌ 核心模块导入失败: {e}")
    sys.exit(1)

# --- 核心配置 ---
WINDOW_WIDTH = 1450
WINDOW_HEIGHT = 900
GRID_SIZE = 42
BOARD_SIZE = 19
START_X = 60
START_Y = 60
HUMAN_MOVE_LIMIT = 30
FIRST_MOVE_TENGEN = True

# 颜色定义
COLORS = {
    'bg': (240, 242, 245),
    'board': (220, 179, 92),
    'grid': (0, 0, 0),
    'black_stone': (15, 15, 15),
    'white_stone': (245, 245, 245),
    'black_panel': (40, 44, 52),
    'white_panel': (255, 255, 255),
    'highlight': (231, 76, 60),
    'active_border': (46, 204, 113),
    'text_dark': (44, 62, 80),
    'text_light': (236, 240, 241),
    'timer_safe': (46, 204, 113),
    'timer_warn': (230, 126, 34),
    'timer_crit': (231, 76, 60),
    'btn_blue': (52, 152, 219),
    'btn_red': (231, 76, 60),
    'btn_gray': (149, 165, 166),
    'btn_green': (39, 174, 96),
    'log_bg': (250, 250, 250),
    'tag_book': (39, 174, 96),
    'tag_ai': (52, 152, 219),
    'modal_overlay': (0, 0, 0, 150),
}

def get_font(size, bold=False):
    font_path = os.path.join(BASE_DIR, "font.ttf")
    if os.path.exists(font_path):
        try: return pygame.font.Font(font_path, size)
        except: pass
    try: return pygame.font.SysFont("simhei", size, bold=bold)
    except: pass
    return pygame.font.Font(None, size)

# --- 工具类 ---
class SymmetryHandler:
    @staticmethod
    def get_canonical_key(board_np):
        flat = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = board_np[r][c]
                flat.append(1 if val == 1 else (2 if val == -1 else 0))
        return tuple(flat)
    @staticmethod
    def get_symmetries(board_np):
        syms = []
        for k in range(4):
            rotated = np.rot90(board_np, k)
            syms.append((rotated, k, False))
            flipped = np.fliplr(rotated)
            syms.append((flipped, k, True))
        return syms
    @staticmethod
    def inverse_coord(r, c, k, is_flipped):
        if is_flipped: c = BOARD_SIZE - 1 - c
        for _ in range(k): r, c = c, BOARD_SIZE - 1 - r
        return r, c

# --- UI 组件 ---
class UIElement:
    def __init__(self, x, y, w, h): self.rect = pygame.Rect(x, y, w, h)
class Button(UIElement):
    def __init__(self, x, y, w, h, text, func, color_theme='btn_blue', font_size=22):
        super().__init__(x, y, w, h); self.text = text; self.func = func
        self.base_color = COLORS.get(color_theme, COLORS['btn_blue'])
        self.hover = False; self.visible = True; self.font = get_font(font_size)
    def draw(self, screen):
        if not self.visible: return
        c = tuple(min(v+20, 255) for v in self.base_color) if self.hover else self.base_color
        pygame.draw.rect(screen, (180,180,180), (self.rect.x, self.rect.y+3, self.rect.w, self.rect.h), border_radius=6)
        pygame.draw.rect(screen, c, self.rect, border_radius=6)
        txt_surf = self.font.render(self.text, True, (255,255,255))
        screen.blit(txt_surf, txt_surf.get_rect(center=self.rect.center))
    def handle_event(self, event):
        if not self.visible: return
        if event.type == pygame.MOUSEMOTION: self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and self.hover: self.func()
class InputBox(UIElement):
    def __init__(self, x, y, w, h, text=''):
        super().__init__(x, y, w, h); self.color = (189, 195, 199); self.text = text; self.active = False; self.font = get_font(24)
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos): self.active = True; self.color = (52, 152, 219); pygame.key.start_text_input(); pygame.key.set_text_input_rect(self.rect)
            else: self.active = False; self.color = (189, 195, 199)
        if self.active:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN: self.active = False; self.color = (189, 195, 199); pygame.key.stop_text_input()
                elif event.key == pygame.K_BACKSPACE: self.text = self.text[:-1]
            elif event.type == pygame.TEXTINPUT: self.text += event.text
    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect, border_radius=5); width = 3 if self.active else 1
        pygame.draw.rect(screen, self.color, self.rect, width, border_radius=5)
        txt_surf = self.font.render(self.text, True, (0,0,0) if self.text else (150,150,150))
        screen.blit(txt_surf, (self.rect.x + 10, self.rect.centery - txt_surf.get_height()//2))
class ModelSelectButton(Button):
    def __init__(self, x, y, w, h, current_model, open_modal_func):
        super().__init__(x, y, w, h, "", open_modal_func, 'btn_gray')
        self.current_model_path = current_model; self.font = get_font(18)
    def update_model(self, path): self.current_model_path = path
    def draw(self, screen):
        pygame.draw.rect(screen, (255,255,255), self.rect, border_radius=5)
        pygame.draw.rect(screen, (52, 152, 219) if self.hover else (189, 195, 199), self.rect, 2, border_radius=5)
        name = os.path.basename(self.current_model_path) if self.current_model_path else "无模型 (纯人操作)"
        color = (0,0,0) if self.current_model_path else (150,150,150)
        txt_surf = self.font.render(name, True, color)
        text_rect = txt_surf.get_rect(midleft=(self.rect.x + 10, self.rect.centery))
        if text_rect.width > self.rect.width - 20: screen.blit(txt_surf, text_rect.topleft, pygame.Rect(0,0,self.rect.width-20, text_rect.height))
        else: screen.blit(txt_surf, text_rect)
class ModelListModal:
    def __init__(self, models, callback_func):
        self.models = [("🚫 不使用模型 (纯人)", None)] + models
        self.callback = callback_func; self.visible = False
        self.rect = pygame.Rect((WINDOW_WIDTH-600)//2, (WINDOW_HEIGHT-500)//2, 600, 500)
        self.close_btn = Button(self.rect.right-40, self.rect.top+10, 30, 30, "X", self.hide, 'btn_red', 18)
    def show(self): self.visible = True
    def hide(self): self.visible = False
    def handle_event(self, event):
        if not self.visible: return False
        self.close_btn.handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if self.rect.collidepoint(mx, my):
                idx = (my - (self.rect.y + 50)) // 50
                if 0 <= idx < len(self.models): self.callback(self.models[idx][1]); self.hide()
        return True
    def draw(self, screen):
        if not self.visible: return
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA); overlay.fill(COLORS['modal_overlay']); screen.blit(overlay, (0,0))
        pygame.draw.rect(screen, (255,255,255), self.rect, border_radius=10)
        pygame.draw.rect(screen, COLORS['btn_blue'], self.rect, 2, border_radius=10)
        screen.blit(get_font(24, True).render("选择AI模型", True, COLORS['text_dark']), (self.rect.x+20, self.rect.y+15))
        self.close_btn.draw(screen)
        for i, (name, _) in enumerate(self.models):
            y = self.rect.y + 50 + i*50
            if y + 40 > self.rect.bottom: break
            bg_col = (240,248,255) if i == 0 else (245,245,245)
            pygame.draw.rect(screen, bg_col, (self.rect.x+10, y, 580, 45), border_radius=5)
            screen.blit(get_font(20).render(name, True, COLORS['text_dark']), (self.rect.x+20, y+10))

# --- 主逻辑 ---

class Connect6GuiFinal:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("智奕连城 - 专业比赛版 (Final)")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_title = get_font(40, True)
        self.font_ui = get_font(20)
        self.font_mono = get_font(18)

        # 1. 加载开局库
        print("正在加载开局库...")
        self.opening_book = {}
        book_path = os.path.join(BASE_DIR, "connect6_book.pkl")
        if os.path.exists(book_path):
            try:
                with open(book_path, "rb") as f:
                    self.opening_book = pickle.load(f)
                print(f"📚 开局库加载成功: {len(self.opening_book)} 个局面")
            except: print("⚠️ 开局库文件损坏")
        else:
            print("⚠️ 未找到 connect6_book.pkl，跳过加载。")

        self.available_models = self.scan_models()
        self.state = 'MENU'

        self.path_b = None
        self.path_w = self.available_models[0][1] if self.available_models else None
        self.white_starts_human = True

        self.init_menu()
        self.game_active = False
        self.winner = None
        self.env = None
        self.agent_b = None
        self.agent_w = None
        self.ai_thinking = False
        self.current_operator = "INIT"
        self.step_timer = 0
        self.move_sources = []

    def scan_models(self):
        patterns = [os.path.join(BASE_DIR, "checkpoints", "*.pt"), os.path.join(BASE_DIR, "*.pt")]
        paths = []
        for p in patterns: paths.extend(glob.glob(p))
        return [(os.path.basename(p), p) for p in paths]

    def init_menu(self):
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        self.input_team_b = InputBox(cx - 300, cy - 120, 200, 40, "黑方战队")
        self.input_team_w = InputBox(cx + 100, cy - 120, 200, 40, "白方战队")
        self.btn_model_b = ModelSelectButton(cx - 300, cy - 40, 200, 40, self.path_b, lambda: self.open_modal('black'))
        self.btn_model_w = ModelSelectButton(cx + 100, cy - 40, 200, 40, self.path_w, lambda: self.open_modal('white'))
        self.btn_w_human = Button(cx + 100, cy + 50, 95, 40, "人类先", lambda: self.set_w_order(True), 'btn_gray', 18)
        self.btn_w_ai = Button(cx + 205, cy + 50, 95, 40, "AI先", lambda: self.set_w_order(False), 'btn_gray', 18)
        self.update_w_order_btns()
        self.btn_start = Button(cx - 100, cy + 150, 200, 60, "开始对抗", self.start_game, 'btn_green', 28)
        self.modal = ModelListModal(self.available_models, self.on_model_selected)
        self.selecting_role = None
        px = START_X + (GRID_SIZE * (BOARD_SIZE - 1)) + 40
        self.btn_undo = Button(px, 800, 100, 45, "悔棋", self.undo_move, 'btn_blue')
        self.btn_menu = Button(px + 120, 800, 100, 45, "菜单", self.back_to_menu, 'btn_gray')
        self.game_buttons = [self.btn_undo, self.btn_menu]
        self.menu_elements = [self.input_team_b, self.input_team_w, self.btn_model_b, self.btn_model_w, self.btn_w_human, self.btn_w_ai, self.btn_start]

    def set_w_order(self, is_human): self.white_starts_human = is_human; self.update_w_order_btns()
    def update_w_order_btns(self):
        self.btn_w_human.base_color = COLORS['btn_blue'] if self.white_starts_human else COLORS['btn_gray']
        self.btn_w_ai.base_color = COLORS['btn_blue'] if not self.white_starts_human else COLORS['btn_gray']
    def open_modal(self, role): self.selecting_role = role; self.modal.show()
    def on_model_selected(self, path):
        if self.selecting_role == 'black': self.path_b = path; self.btn_model_b.update_model(path)
        else: self.path_w = path; self.btn_model_w.update_model(path)

    def load_agent(self, path):
        if not path: return None
        print(f"加载模型: {path}")
        m_type = "hybrid" if "hybrid" in os.path.basename(path).lower() else "advanced"
        print(f"模型类型: {m_type}")
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            sims = 200 if device == 'cuda' else 50
            return Connect6Player(model_path=path, num_simulations=sims, device=device)
        except: return None

    def get_current_operator_type(self):
        if self.env.game_over: return "NONE"
        idx = len(self.env.last_moves)
        turn_num = 1 if idx == 0 else (idx + 3) // 2
        is_black_turn = (turn_num % 2 == 1)

        rule_role = "Human"
        if is_black_turn:
            cycle = (turn_num - 1) // 2
            rule_role = "Human" if cycle % 2 == 0 else "AI"
        else:
            cycle = (turn_num - 2) // 2
            rule_role = ("Human" if cycle % 2 == 0 else "AI") if self.white_starts_human else ("AI" if cycle % 2 == 0 else "Human")

        if rule_role == "AI":
            agent = self.agent_b if is_black_turn else self.agent_w
            if agent is None: return "Human"
            return "AI"
        return "Human"

    def start_game(self):
        self.agent_b = self.load_agent(self.path_b)
        self.agent_w = self.load_agent(self.path_w)
        self.env = Connect6Env()
        self.env.reset()
        self.game_active = True
        self.winner = None
        self.state = 'GAME'
        self.current_step_start_time = time.time()
        self.current_operator = self.get_current_operator_type()
        self.move_sources = []

    def back_to_menu(self): self.state = 'MENU'; self.game_active = False

    def undo_move(self):
        if self.ai_thinking or len(self.env.last_moves) == 0: return
        r, c = self.env.last_moves.pop()
        self.env.board[r][c] = 0
        if self.move_sources: self.move_sources.pop()
        self.winner = None; self.game_active = True

        cnt = len(self.env.last_moves)
        if cnt == 0: self.env.current_player = Player.BLACK
        elif cnt == 1: self.env.current_player = Player.WHITE
        else:
            turn_num = (cnt + 3) // 2
            self.env.current_player = Player.BLACK if (turn_num % 2 == 1) else Player.WHITE

        self.current_operator = self.get_current_operator_type()
        self.current_step_start_time = time.time()

    def check_opening_book_moves(self):
        if not self.opening_book: return False, []
        current_board_np = np.array(self.env.board)
        symmetries = SymmetryHandler.get_symmetries(current_board_np)
        for transformed_board, k, is_flipped in symmetries:
            key = SymmetryHandler.get_canonical_key(transformed_board)
            if key in self.opening_book:
                book_moves = self.opening_book[key]
                if len(book_moves) > 0:
                    real_moves = []
                    for tr, tc in book_moves:
                        orig_r, orig_c = SymmetryHandler.inverse_coord(tr, tc, k, is_flipped)
                        if self.env.is_valid_move(orig_r, orig_c):
                            real_moves.append((orig_r, orig_c))
                    if real_moves:
                        return True, real_moves
        return False, []

    def ai_worker(self, agent):
        try:
            next_move_coords = None
            source_tag = "AI"

            # 强制天元
            if FIRST_MOVE_TENGEN and len(self.env.last_moves) == 0:
                print("✨ [定式] 强制开局: 天元 (J,10)")
                next_move_coords = (9, 9)
                source_tag = "Book"

            # 每下一子后重新查表；同组坐标为等价候选。
            if next_move_coords is None:
                found, moves = self.check_opening_book_moves()
                if found and moves:
                    print(f"📖 [定式] 命中 {len(moves)} 个候选，选择一子")
                    r, c = moves[0]
                    if self.env.is_valid_move(r, c):
                        next_move_coords = (r, c)
                        source_tag = "Book"

            if next_move_coords:
                action = self.env.coord_to_action(*next_move_coords)
                time.sleep(0.5)
            else:
                print("🧠 [AI] 思考中...")
                action = agent.get_action(self.env)
                source_tag = "AI"

            if self.game_active:
                self.env.step(action)
                self.post_move(source_tag)

        except: traceback.print_exc()
        finally: self.ai_thinking = False

    def post_move(self, source="Human"):
        self.move_sources.append(source)
        if self.env.game_over:
            self.save_record(self.env.winner, "连成6子")
            self.winner = self.env.winner
            self.game_active = False
        else:
            self.current_step_start_time = time.time()
            self.current_operator = self.get_current_operator_type()

    def save_record(self, winner, reason):
        # 1. 使用绝对路径
        record_dir = os.path.join(BASE_DIR, 'records')
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)

        # 2. 清洗非法字符
        t1 = "".join([c for c in self.input_team_b.text if c.isalnum() or c in (' ', '_', '-')]).strip()
        t2 = "".join([c for c in self.input_team_w.text if c.isalnum() or c in (' ', '_', '-')]).strip()
        if not t1: t1 = "Black"
        if not t2: t2 = "White"

        res = "BlackWin" if winner == Player.BLACK else "WhiteWin"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{t1}_vs_{t2}_{res}.txt"
        filepath = os.path.join(record_dir, filename)

        try:
            with open(filepath, "w", encoding='utf-8') as f:
                f.write(f"Black: {t1}\nWhite: {t2}\nResult: {res}\nReason: {reason}\nMoves:\n")
                for i, (r, c) in enumerate(self.env.last_moves):
                    f.write(f"{i+1}. {chr(ord('A')+c)}{19-r}\n")
            print(f"✅ 棋谱已保存: {filename}")
        except Exception as e:
            print(f"❌ 棋谱保存失败: {e}")

    def update(self):
        if not self.game_active or self.winner: return
        self.step_timer = time.time() - self.current_step_start_time

        if self.current_operator == "Human":
            if self.step_timer > HUMAN_MOVE_LIMIT:
                idx = len(self.env.last_moves)
                is_black = (idx == 0 or ((idx+3)//2 % 2 == 1))
                winner = Player.WHITE if is_black else Player.BLACK
                self.winner = winner; self.game_active = False
                self.save_record(winner, "超时")

        elif self.current_operator == "AI" and not self.ai_thinking:
            self.ai_thinking = True
            idx = len(self.env.last_moves)
            is_black = (idx == 0 or ((idx+3)//2 % 2 == 1))
            agent = self.agent_b if is_black else self.agent_w
            threading.Thread(target=self.ai_worker, args=(agent,), daemon=True).start()

    def handle_click(self, pos):
        if not self.game_active or self.winner or self.ai_thinking: return
        if self.current_operator != "Human": return
        c, r = round((pos[0]-START_X)/GRID_SIZE), round((pos[1]-START_Y)/GRID_SIZE)
        if 0<=r<BOARD_SIZE and 0<=c<BOARD_SIZE:
             if (pos[0]-(START_X+c*GRID_SIZE))**2 + (pos[1]-(START_Y+r*GRID_SIZE))**2 < 400:
                 if self.env.is_valid_move(r, c):
                     self.env.step(self.env.coord_to_action(r, c))
                     self.post_move("Human")

    def draw(self):
        self.screen.fill(COLORS['bg'])
        if self.state == 'MENU':
            self.screen.blit(self.font_title.render("智奕连城 - 专业版", True, COLORS['text_dark']), (WINDOW_WIDTH//2-200, 80))
            cx, cy = WINDOW_WIDTH//2, WINDOW_HEIGHT//2
            self.screen.blit(self.font_ui.render("黑方配置", True, COLORS['text_dark']), (cx-300, cy-160))
            self.screen.blit(self.font_ui.render("白方配置", True, COLORS['text_dark']), (cx+100, cy-160))
            for e in self.menu_elements: e.draw(self.screen)
            self.modal.draw(self.screen)
        else:
            pygame.draw.rect(self.screen, COLORS['board'], (START_X-20, START_Y-20, 796, 796), border_radius=5)
            for i in range(19):
                s, e = START_X+i*GRID_SIZE, START_Y+i*GRID_SIZE
                pygame.draw.line(self.screen, COLORS['grid'], (START_X, e), (START_X+756, e)); pygame.draw.line(self.screen, COLORS['grid'], (s, START_Y), (s, START_Y+756))
                self.screen.blit(get_font(16).render(str(19-i), True, COLORS['text_dark']), (START_X-35, e-8))
                self.screen.blit(get_font(16).render(chr(ord('A')+i), True, COLORS['text_dark']), (s-5, START_Y+765))
            for r in [3,9,15]:
                for c in [3,9,15]: pygame.draw.circle(self.screen, COLORS['grid'], (START_X+c*GRID_SIZE, START_Y+r*GRID_SIZE), 4)
            for i, (r,c) in enumerate(self.env.last_moves):
                ib = (i==0 or ((i+3)//2)%2==1)
                col = COLORS['black_stone'] if ib else COLORS['white_stone']
                px, py = START_X+c*GRID_SIZE, START_Y+r*GRID_SIZE
                pygame.draw.circle(self.screen, col, (px, py), 19)
                if not ib: pygame.draw.circle(self.screen, (100,100,100), (px, py), 19, 1)
                if i==len(self.env.last_moves)-1: pygame.draw.circle(self.screen, COLORS['highlight'], (px, py), 6)
                self.screen.blit(get_font(14).render(str(i+1), True, COLORS['highlight'] if not ib else (200,200,200)), (px-5, py-7))

            px = START_X+800
            idx = len(self.env.last_moves)
            is_black_active = (idx==0 or ((idx+3)//2)%2==1) if not self.winner else False

            pygame.draw.rect(self.screen, COLORS['black_panel'], (px, START_Y, 280, 160), border_radius=8)
            if is_black_active: pygame.draw.rect(self.screen, COLORS['active_border'], (px, START_Y, 280, 160), 3, border_radius=8)
            self.screen.blit(get_font(24, True).render(f"⚫ {self.input_team_b.text}", True, COLORS['text_light']), (px+15, START_Y+15))
            role_b = "混合 AI" if self.path_b else "纯人操作"
            self.screen.blit(get_font(18).render(role_b, True, (200,200,200)), (px+15, START_Y+50))
            if self.winner==Player.BLACK: st="获胜!"; cl=COLORS['highlight']
            elif is_black_active:
                if self.current_operator=="Human":
                    rule_ai = self.path_b and (((idx+3)//2-1)//2)%2==1
                    st = "请人工代打" if rule_ai else f"落子: {30-self.step_timer:.1f}s"
                    cl = COLORS['timer_warn']
                else: st="AI 思考中..."; cl=COLORS['btn_blue']
            else: st="等待"; cl=(100,100,100)
            self.screen.blit(get_font(28, True).render(st, True, cl), (px+15, START_Y+90))

            py = START_Y+180
            pygame.draw.rect(self.screen, COLORS['white_panel'], (px, py, 280, 160), border_radius=8)
            if not is_black_active and not self.winner: pygame.draw.rect(self.screen, COLORS['active_border'], (px, py, 280, 160), 3, border_radius=8)
            self.screen.blit(get_font(24, True).render(f"⚪ {self.input_team_w.text}", True, COLORS['text_dark']), (px+15, py+15))
            role_w = "混合 AI" if self.path_w else "纯人操作"
            self.screen.blit(get_font(18).render(role_w, True, COLORS['text_dark']), (px+15, py+50))
            if self.winner==Player.WHITE: st="获胜!"; cl=COLORS['highlight']
            elif not is_black_active and not self.winner:
                 if self.current_operator=="Human":
                    st = f"落子: {30-self.step_timer:.1f}s"; cl = COLORS['timer_warn']
                 else: st="AI 思考中..."; cl=COLORS['btn_blue']
            else: st="等待"; cl=COLORS['text_dark']
            self.screen.blit(get_font(28, True).render(st, True, cl), (px+15, py+90))

            py_l = py+180
            pygame.draw.rect(self.screen, COLORS['log_bg'], (px, py_l, 280, 350))
            pygame.draw.rect(self.screen, (200,200,200), (px, py_l, 280, 350), 1)
            surf = pygame.Surface((270, 320)); surf.fill(COLORS['log_bg'])
            off = max(0, len(self.env.last_moves)*22 - 300)
            for i, (r,c) in enumerate(self.env.last_moves):
                y = i*22 - off
                if -22<y<320:
                    ib = (i==0 or ((i+3)//2)%2==1)
                    src = self.move_sources[i] if i < len(self.move_sources) else ""
                    txt = f"{i+1:02d}. {'B' if ib else 'W'}({chr(ord('A')+c)},{19-r})"
                    surf.blit(self.font_mono.render(txt, True, (0,0,0) if ib else (80,80,100)), (5, y))
                    if src == "Book": surf.blit(get_font(14).render("[定式]", True, COLORS['tag_book']), (160, y+2))
                    elif src == "AI": surf.blit(get_font(14).render("[AI]", True, COLORS['tag_ai']), (160, y+2))
            self.screen.blit(surf, (px+5, py_l+25))
            for b in self.game_buttons: b.draw(self.screen)

    def run(self):
        while True:
            if self.state=='GAME': self.update()
            for e in pygame.event.get():
                if e.type==pygame.QUIT: pygame.quit(); sys.exit()
                if self.state=='MENU':
                    if self.modal.visible: self.modal.handle_event(e)
                    else:
                        for el in self.menu_elements: el.handle_event(e)
                else:
                    for b in self.game_buttons: b.handle_event(e)
                    if e.type==pygame.MOUSEBUTTONDOWN: self.handle_click(e.pos)
            self.draw()
            pygame.display.flip(); self.clock.tick(30)

if __name__ == "__main__":
    Connect6GuiFinal().run()
