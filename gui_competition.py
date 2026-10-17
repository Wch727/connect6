#!/usr/bin/env python3
"""
智奕连城 - 比赛通用客户端 (V8.9 UI调整版)
---------------------------------------------------------
更新日志:
1. [UI] 将“获胜/状态”文字块移至右侧面板最顶部 (y=15)，并放大加粗，一目了然。
2. [UI] 保持 "打开目录" 按钮移除状态，界面清爽。
3. [Core] 继承 V8.8 所有智能策略与存档保护机制。
"""

import pygame
import sys
import os
import time
import threading
import traceback
import glob
import pickle
import random
import re
import numpy as np
from datetime import datetime

# --- 路径修正 ---
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print("="*60)
print(f"📂 程序运行根目录: {BASE_DIR}")
print(f"💾 棋谱将保存在:   {os.path.join(BASE_DIR, 'records')}")
print("="*60)

try:
    from game.connect6_env import Connect6Env, Player
    from play import Connect6Player
except ImportError as e:
    print(f"❌ 导入失败: {e}\n请确保 game/ 和 play.py 在 {BASE_DIR} 目录下。")
    sys.exit(1)

# --- 核心配置 ---
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
GRID_SIZE = 42
BOARD_SIZE = 19
START_X = 80
START_Y = 60
TIME_LIMIT = 300000

# [策略配置]
BOOK_ACCEPT_RATE = 1.0
FIRST_MOVE_TENGEN_RATE = 0.8

COLORS = {
    'bg': (240, 242, 245),
    'board': (220, 179, 92),
    'grid': (0, 0, 0),
    'black': (15, 15, 15),
    'white': (245, 245, 245),
    'highlight': (231, 76, 60),
    'text': (44, 62, 80),
    'sub_text': (100, 100, 100),
    'timer_normal': (46, 204, 113),
    'timer_low': (231, 76, 60),
    'btn_blue': (52, 152, 219),
    'btn_red': (231, 76, 60),
    'btn_gray': (149, 165, 166),
    'btn_green': (39, 174, 96),
    'input_bg': (255, 255, 255),
    'input_border': (189, 195, 199),
    'input_active': (52, 152, 219),
    'input_text': (0, 0, 0),
    'panel_bg': (255, 255, 255),
    'modal_overlay': (0, 0, 0, 150),
    'modal_bg': (255, 255, 255),
    'log_text': (60, 60, 60),
    'tag_book': (39, 174, 96),
    'tag_ai': (52, 152, 219),
    'tag_rand': (230, 126, 34)
}

def get_font(size, bold=False):
    font_path = os.path.join(BASE_DIR, "font.ttf")
    if os.path.exists(font_path):
        try:
            return pygame.font.Font(font_path, size)
        except:
            pass
    try: return pygame.font.SysFont("simhei", size, bold=bold)
    except: pass
    try: return pygame.font.SysFont("microsoftyahei", size, bold=bold)
    except: pass
    return pygame.font.Font(None, size)

class SymmetryHandler:
    @staticmethod
    def get_canonical_key(board_np):
        flat = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = board_np[r][c]
                if val == 1: flat.append(1)
                elif val == -1: flat.append(2)
                elif val == 2: flat.append(2)
                else: flat.append(0)
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
        if is_flipped:
            c = BOARD_SIZE - 1 - c
        for _ in range(k):
            r, c = c, BOARD_SIZE - 1 - r
        return r, c

# --- UI 组件库 ---
class UIElement:
    def __init__(self, x, y, w, h): self.rect = pygame.Rect(x, y, w, h)
class Button(UIElement):
    def __init__(self, x, y, w, h, text, func, color_theme='btn_blue', font_size=24):
        super().__init__(x, y, w, h); self.text = text; self.func = func; self.base_color = COLORS.get(color_theme, COLORS['btn_blue']); self.hover = False; self.visible = True; self.font = get_font(font_size)
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
    def __init__(self, x, y, w, h, text=''): super().__init__(x, y, w, h); self.color = COLORS['input_border']; self.text = text; self.active = False; self.font = get_font(24)
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos): self.active = True; self.color = COLORS['input_active']; pygame.key.start_text_input(); pygame.key.set_text_input_rect(self.rect)
            else: self.active = False; self.color = COLORS['input_border']
        if self.active:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN: self.active = False; self.color = COLORS['input_border']; pygame.key.stop_text_input()
                elif event.key == pygame.K_BACKSPACE: self.text = self.text[:-1]
            elif event.type == pygame.TEXTINPUT: self.text += event.text
    def draw(self, screen):
        pygame.draw.rect(screen, COLORS['input_bg'], self.rect, border_radius=5); width = 3 if self.active else 1
        pygame.draw.rect(screen, self.color, self.rect, width, border_radius=5)
        display_text = self.text if self.text else "点击输入..."
        color = COLORS['input_text'] if self.text else COLORS['sub_text']
        txt_surf = self.font.render(display_text, True, color)
        text_rect = txt_surf.get_rect(midleft=(self.rect.x + 10, self.rect.centery))
        if text_rect.width > self.rect.width - 20: screen.blit(txt_surf, (self.rect.x + 10, text_rect.y), pygame.Rect(text_rect.width - (self.rect.width - 20), 0, self.rect.width - 20, text_rect.height))
        else: screen.blit(txt_surf, text_rect)
class ModelSelectButton(Button):
    def __init__(self, x, y, w, h, current_model, open_modal_func): super().__init__(x, y, w, h, "", open_modal_func, 'input_bg'); self.current_model_path = current_model; self.font = get_font(20); self.text_color = COLORS['text']
    def update_model(self, path): self.current_model_path = path
    def draw(self, screen):
        pygame.draw.rect(screen, COLORS['input_bg'], self.rect, border_radius=5); color = COLORS['input_active'] if self.hover else COLORS['input_border']; pygame.draw.rect(screen, color, self.rect, 2, border_radius=5)
        name = "不使用模型 (人类)";
        if self.current_model_path: name = os.path.basename(self.current_model_path)
        txt_surf = self.font.render(name, True, self.text_color); text_rect = txt_surf.get_rect(midleft=(self.rect.x + 10, self.rect.centery))
        if text_rect.width > self.rect.width - 30: screen.blit(txt_surf, text_rect.topleft, pygame.Rect(0, 0, self.rect.width - 30, text_rect.height))
        else: screen.blit(txt_surf, text_rect)
        pygame.draw.polygon(screen, (100,100,100), [(self.rect.right - 20, self.rect.centery - 3), (self.rect.right - 10, self.rect.centery - 3), (self.rect.right - 15, self.rect.centery + 3)])
class ModelListModal:
    def __init__(self, models, callback_func):
        self.models = [("不使用模型 (人类)", None)] + models; self.callback = callback_func; self.visible = False; self.w, self.h = 600, 500; self.x = (WINDOW_WIDTH - self.w) // 2; self.y = (WINDOW_HEIGHT - self.h) // 2; self.rect = pygame.Rect(self.x, self.y, self.w, self.h); self.scroll_y = 0; self.item_h = 50; self.font = get_font(22); self.close_btn = Button(self.x + self.w - 40, self.y + 10, 30, 30, "X", self.hide, 'btn_red', 18)
    def show(self): self.visible = True; self.scroll_y = 0
    def hide(self): self.visible = False
    def handle_event(self, event):
        if not self.visible: return False
        self.close_btn.handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4: self.scroll_y = min(0, self.scroll_y + 30)
            elif event.button == 5: max_scroll = -(len(self.models) * self.item_h - (self.h - 60)); self.scroll_y = max(max_scroll, self.scroll_y - 30) if max_scroll < 0 else 0
            elif event.button == 1:
                mx, my = event.pos; list_rect = pygame.Rect(self.x, self.y + 50, self.w, self.h - 60)
                if list_rect.collidepoint(mx, my):
                    idx = (my - (self.y + 50) - self.scroll_y) // self.item_h
                    if 0 <= idx < len(self.models): self.callback(self.models[idx][1]); self.hide()
        return True
    def draw(self, screen):
        if not self.visible: return
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA); overlay.fill(COLORS['modal_overlay']); screen.blit(overlay, (0,0))
        pygame.draw.rect(screen, COLORS['modal_bg'], self.rect, border_radius=10); pygame.draw.rect(screen, COLORS['btn_blue'], self.rect, 2, border_radius=10)
        title = get_font(24, True).render("请选择 AI 模型文件", True, COLORS['text']); screen.blit(title, (self.x + 20, self.y + 15)); self.close_btn.draw(screen)
        list_surface = pygame.Surface((self.w - 20, self.h - 60)); list_surface.fill(COLORS['modal_bg']); mx, my = pygame.mouse.get_pos(); local_mx, local_my = mx - (self.x + 10), my - (self.y + 50)
        for i, (name, path) in enumerate(self.models):
            y_pos = i * self.item_h + self.scroll_y
            if -self.item_h < y_pos < self.h:
                rect = pygame.Rect(0, y_pos, self.w - 20, self.item_h - 5)
                if rect.collidepoint(local_mx, local_my): pygame.draw.rect(list_surface, (230, 240, 255), rect, border_radius=5)
                else: pygame.draw.rect(list_surface, (245, 245, 245), rect, border_radius=5)
                txt = self.font.render(name, True, COLORS['text']); list_surface.blit(txt, (10, y_pos + 12))
                if path: path_txt = get_font(14).render(path[-45:], True, COLORS['sub_text']); list_surface.blit(path_txt, (10, y_pos + 32))
        screen.blit(list_surface, (self.x + 10, self.y + 50))

# --- 主程序 ---

class Connect6GUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("智奕连城 - 比赛系统 V8.9 (UI调整版)")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_title = get_font(48, True)
        self.font_label = get_font(24)
        self.font_mono = get_font(16)

        print("正在加载开局库...")
        self.opening_book = {}
        book_path = os.path.join(BASE_DIR, "connect6_book.pkl")
        if os.path.exists(book_path):
            try:
                with open(book_path, "rb") as f:
                    self.opening_book = pickle.load(f)
                print(f"📚 开局库加载成功: {len(self.opening_book)} 个局面")
            except Exception as e:
                print(f"⚠️ 开局库加载失败: {e}")
        else:
            print(f"⚠️ 未找到 connect6_book.pkl (路径:{book_path})，将不使用开局库。")

        self.available_models = self.scan_models()
        self.state = 'MENU'
        self.menu_ui_elements = self.init_menu_ui()
        self.game_ui_elements = []
        self.modal = ModelListModal(self.available_models, self.on_model_selected)

        self.current_selecting_role = None
        self.env = None
        self.agent_black = None
        self.agent_white = None
        self.team_b_name = ""
        self.team_w_name = ""
        self.path_b = None
        self.path_w = None

        if self.available_models:
            self.path_w = self.available_models[0][1]
            self.menu_ui_elements[3].update_model(self.path_w)

        self.game_active = False
        self.winner = None
        self.ai_thinking = False
        self.ai_action_queue = None
        self.ai_current_source = None
        self.move_sources = []
        self.has_saved = False
        self.save_status_msg = ""

        self.time_b = TIME_LIMIT
        self.time_w = TIME_LIMIT
        self.last_time = 0
        self.replay_idx = -1

    def scan_models(self):
        patterns = [
            os.path.join(BASE_DIR, "checkpoints", "*.pt"),
            os.path.join(BASE_DIR, "*.pt"),
            os.path.join(os.getcwd(), "checkpoints", "*.pt")
        ]
        paths = []
        for p in patterns: paths.extend(glob.glob(p))
        paths = list(set([os.path.abspath(p) for p in paths]))
        paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return [(os.path.basename(p), p) for p in paths]

    def init_menu_ui(self):
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        return [
            InputBox(cx - 350, cy - 60, 250, 45, "Player1"),
            ModelSelectButton(cx - 350, cy + 10, 250, 45, None, lambda: self.open_modal('black')),
            InputBox(cx + 100, cy - 60, 250, 45, "AI_Bot"),
            ModelSelectButton(cx + 100, cy + 10, 250, 45, None, lambda: self.open_modal('white')),
            Button(cx - 100, cy + 150, 200, 60, "开始比赛", self.start_game, 'btn_green', 28)
        ]

    def init_game_ui(self):
        px, py, h = START_X + (GRID_SIZE * (BOARD_SIZE - 1)) + 50, 520, 50
        self.game_ui_elements = [
            Button(px, py, 140, h, "悔棋", self.undo_move, 'btn_gray'),
            Button(px+160, py, 140, h, "重置", self.reset_game, 'btn_red'),
            Button(px, py+70, 140, h, "< 上一步", self.replay_prev, 'btn_blue'),
            Button(px+160, py+70, 140, h, "下一步 >", self.replay_next, 'btn_blue'),
            Button(px+80, py+140, 140, h, "返回菜单", self.back_to_menu, 'btn_gray')
        ]
        self.game_ui_elements[2].visible = False
        self.game_ui_elements[3].visible = False

    def open_modal(self, role):
        self.current_selecting_role = role
        self.modal.show()

    def on_model_selected(self, path):
        if self.current_selecting_role == 'black':
            self.path_b = path
            self.menu_ui_elements[1].update_model(path)
        else:
            self.path_w = path
            self.menu_ui_elements[3].update_model(path)

    def load_agent(self, path):
        if not path: return None
        print(f"加载模型: {path}")
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            sims = 3000 if device == 'cuda' else 50
            return Connect6Player(model_path=path, num_simulations=sims, device=device)
        except:
            traceback.print_exc()
            return None

    def start_game(self):
        self.team_b_name = self.menu_ui_elements[0].text
        self.team_w_name = self.menu_ui_elements[2].text
        self.agent_black = self.load_agent(self.path_b)
        self.agent_white = self.load_agent(self.path_w)
        self.env = Connect6Env()
        self.reset_game_logic()
        self.init_game_ui()
        self.state = 'GAME'

    def reset_game(self):
        self.env.reset()
        self.reset_game_logic()

    def reset_game_logic(self):
        self.game_active = True
        self.winner = None
        self.ai_thinking = False
        self.ai_action_queue = None
        self.ai_current_source = None
        self.move_sources = []
        self.has_saved = False
        self.save_status_msg = ""
        self.time_b = TIME_LIMIT
        self.time_w = TIME_LIMIT
        self.last_time = time.time()
        self.replay_idx = -1

    def back_to_menu(self):
        self.state = 'MENU'

    def update_timer(self):
        if not self.game_active or self.winner: return
        now = time.time(); dt = now - self.last_time; self.last_time = now
        if self.env.current_player == Player.BLACK: self.time_b -= dt
        else: self.time_w -= dt
        if self.time_b <= 0: self.finish_game(Player.WHITE, "超时")
        elif self.time_w <= 0: self.finish_game(Player.BLACK, "超时")

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

    def check_ai(self):
        if not self.game_active or self.winner or self.ai_thinking: return
        agent = self.agent_black if self.env.current_player == Player.BLACK else self.agent_white
        if agent:
            self.ai_thinking = True
            threading.Thread(target=self.ai_worker, args=(agent,), daemon=True).start()

    def ai_worker(self, agent):
        try:
            # 1. 第一手
            stone_count = np.count_nonzero(self.env.board)
            if stone_count == 0:
                if random.random() < FIRST_MOVE_TENGEN_RATE:
                    print("✨ [定式] 强制开局: 天元 (J,10)")
                    r, c = 9, 9
                    self.ai_current_source = "Book"
                else:
                    r = random.randint(7, 11)
                    c = random.randint(7, 11)
                    print(f"🎲 [随机] 开局: ({self.get_log_coord(r, c)})")
                    self.ai_current_source = "Rand"

                self.ai_action_queue = self.env.coord_to_action(r, c)
                return

            # 2. 查表
            found, move_coords_list = self.check_opening_book_moves()
            use_book = False
            if found and move_coords_list:
                if random.random() < BOOK_ACCEPT_RATE: use_book = True
                else: print("🎲 [策略] 即使命中定式，AI决定主动弃用，进行深度思考...")

            if use_book:
                r, c = move_coords_list[0]
                if self.env.board[r][c] == 0:
                    print(f"📖 [定式] 命中 {len(move_coords_list)} 个候选，选择一子")
                    self.ai_action_queue = self.env.coord_to_action(r, c)
                    self.ai_current_source = "Book"
                    return

            # 3. 计算
            print("🧠 [AI] 思考中...")
            action = agent.get_action(self.env)
            self.ai_action_queue = action
            self.ai_current_source = "AI"
        except:
            traceback.print_exc()
            self.ai_thinking = False

    def process_ai(self):
        if self.ai_thinking and self.ai_action_queue is not None:
            self.env.step(self.ai_action_queue)
            if self.ai_current_source: self.move_sources.append(self.ai_current_source)
            else: self.move_sources.append("Unknown")
            self.ai_thinking = False
            self.ai_action_queue = None
            self.ai_current_source = None
            if self.env.game_over: self.finish_game(self.env.winner)

    def finish_game(self, winner, reason=""):
        if not self.game_active: return
        self.winner = winner
        self.game_active = False
        self.ai_thinking = False
        self.auto_save_log()
        self.replay_idx = -1
        self.game_ui_elements[2].visible = True
        self.game_ui_elements[3].visible = True

    def undo_move(self):
        if self.ai_thinking or not self.game_active: return
        if not self.env.last_moves: return
        moves = self.env.last_moves[:-1]
        self.env.reset()
        self.move_sources = self.move_sources[:len(moves)]
        for r, c in moves: self.env.step(self.env.coord_to_action(r, c))
        self.replay_idx = -1

    def replay_prev(self):
        if not self.env.last_moves: return
        total = len(self.env.last_moves)
        if self.replay_idx == -1: self.replay_idx = total
        if self.replay_idx > 0: self.replay_idx -= 1

    def replay_next(self):
        if self.replay_idx == -1: return
        total = len(self.env.last_moves)
        if self.replay_idx < total: self.replay_idx += 1
        if self.replay_idx == total: self.replay_idx = -1

    def get_log_coord(self, r, c): return f"{chr(ord('A') + c)},{19 - r}"

    def sanitize_filename(self, name):
        return re.sub(r'[\\/*?:"<>|]', '_', name)

    def auto_save_log(self):
        if self.has_saved or not self.env.last_moves: return

        if self.winner == Player.BLACK: res_str = "先手胜"; file_res = "先手胜"
        elif self.winner == Player.WHITE: res_str = "后手胜"; file_res = "后手胜"
        else: res_str = "和棋"; file_res = "和棋"

        ts = datetime.now().strftime("%Y.%m.%d %H:%M")

        raw_n1 = self.team_b_name.strip() or "Player1"
        raw_n2 = self.team_w_name.strip() or "AI_Bot"

        n1_safe = self.sanitize_filename(raw_n1)
        n2_safe = self.sanitize_filename(raw_n2)

        records_dir = os.path.join(BASE_DIR, "records")
        os.makedirs(records_dir, exist_ok=True)

        filename = f"C6-{n1_safe} vs {n2_safe}-{file_res}.txt"
        full_path = os.path.join(records_dir, filename)

        moves_str_list = []
        for i, (r,c) in enumerate(self.env.last_moves):
            color_char = 'B' if i==0 or ((i+1)//2)%2==0 else 'W'
            coord = self.get_log_coord(r,c)
            moves_str_list.append(f"{color_char}({coord})")

        ct = f"{{[C6][{raw_n1}][{raw_n2}][{res_str}][{ts} Local][Beijing_AI_Contest];\n{';'.join(moves_str_list)}}}"

        try:
            with open(full_path, "w", encoding="gb2312") as f:
                f.write(ct)
            print(f"✅ 保存成功: {full_path}")
            self.save_status_msg = f"✅ 棋谱已保存: records/{filename}"
        except Exception as e:
            print(f"⚠️ GB2312 失败，尝试 GBK: {e}")
            try:
                with open(full_path, "w", encoding="gbk") as f:
                    f.write(ct)
                print(f"✅ GBK 保存成功")
                self.save_status_msg = f"✅ (GBK)已保存: records/{filename}"
            except Exception as e2:
                print(f"❌ 保存彻底失败: {e2}")
                self.save_status_msg = f"❌ 保存失败: {str(e2)[:20]}..."

        self.has_saved = True

    def coord_to_pixel(self, r, c): return START_X + c * GRID_SIZE, START_Y + r * GRID_SIZE
    def pixel_to_coord(self, px, py):
        c, r = round((px-START_X)/GRID_SIZE), round((py-START_Y)/GRID_SIZE)
        if 0<=r<BOARD_SIZE and 0<=c<BOARD_SIZE:
            cx, cy = self.coord_to_pixel(r, c)
            if (px-cx)**2+(py-cy)**2 < (GRID_SIZE/2)**2: return r, c
        return None

    def draw_menu(self):
        self.screen.fill(COLORS['bg']); cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        title = self.font_title.render("智奕连城 - 比赛配置", True, COLORS['text']); self.screen.blit(title, title.get_rect(center=(cx, 100)))
        pygame.draw.line(self.screen, (200,200,200), (cx, 180), (cx, 600), 2)
        lbl_b = self.font_label.render("黑方 (先手)", True, COLORS['black']); self.screen.blit(lbl_b, (cx - 450, cy - 100))
        self.menu_ui_elements[0].draw(self.screen); self.menu_ui_elements[1].draw(self.screen)
        lbl_w = self.font_label.render("白方 (后手)", True, COLORS['text']); self.screen.blit(lbl_w, (cx + 100, cy - 100))
        self.menu_ui_elements[2].draw(self.screen); self.menu_ui_elements[3].draw(self.screen); self.menu_ui_elements[4].draw(self.screen)
        if self.modal.visible: self.modal.draw(self.screen)

    def draw_move_list(self):
        px = START_X + (GRID_SIZE * (BOARD_SIZE - 1)) + 50; py = 320; w, h = 300, 180
        pygame.draw.rect(self.screen, COLORS['panel_bg'], (px, py, w, h), border_radius=8); pygame.draw.rect(self.screen, (200,200,200), (px, py, w, h), 2, border_radius=8)
        self.screen.blit(get_font(16).render("实时棋谱 / 复盘", True, COLORS['text']), (px+10, py+5))
        pygame.draw.line(self.screen, (220,220,220), (px+5, py+25), (px+w-5, py+25))
        moves = self.env.last_moves
        if self.replay_idx != -1: moves = self.env.last_moves[:self.replay_idx]
        display_end = len(moves); start_idx = max(0, display_end - 8)
        for i in range(start_idx, display_end):
            r, c = moves[i]; is_black = (i == 0) or (((i+1)//2)%2 == 0)
            coord_text = f"{i+1:02d}. {'B' if is_black else 'W'}({self.get_log_coord(r, c)})"
            tag = ""; tag_color = COLORS['sub_text']
            if i < len(self.move_sources):
                src = self.move_sources[i]
                if src == "Book": tag = " [定式]"; tag_color = COLORS['tag_book']
                elif src == "AI": tag = " [AI]"; tag_color = COLORS['tag_ai']
                elif src == "Rand": tag = " [随机]"; tag_color = COLORS['tag_rand']
            base_surf = self.font_mono.render(coord_text, True, COLORS['log_text']); self.screen.blit(base_surf, (px+20, py+30+(i-start_idx)*18))
            if tag: tag_surf = self.font_mono.render(tag, True, tag_color); self.screen.blit(tag_surf, (px+20 + base_surf.get_width(), py+30+(i-start_idx)*18))

    def draw_game(self):
        self.screen.fill(COLORS['bg'])
        pygame.draw.rect(self.screen, (200,200,200), (START_X-24, START_Y-24, 756+48, 756+48), border_radius=5); pygame.draw.rect(self.screen, COLORS['board'], (START_X-20, START_Y-20, 756+40, 756+40), border_radius=5)
        for i in range(19):
            y, x = START_Y + i * GRID_SIZE, START_X + i * GRID_SIZE
            pygame.draw.line(self.screen, COLORS['grid'], (START_X, y), (START_X + 756, y), 1); pygame.draw.line(self.screen, COLORS['grid'], (x, START_Y), (x, START_Y + 756), 1)
            self.screen.blit(get_font(16).render(str(19-i), True, COLORS['text']), (START_X-30, y-8)); self.screen.blit(get_font(16).render(chr(ord('A')+i), True, COLORS['text']), (x-5, START_Y+756+10))
        for r in [3,9,15]:
            for c in [3,9,15]: pygame.draw.circle(self.screen, COLORS['grid'], self.coord_to_pixel(r, c), 4)
        moves = self.env.last_moves
        if self.replay_idx != -1: moves = self.env.last_moves[:self.replay_idx]
        for i, (r, c) in enumerate(moves):
            color = COLORS['black'] if (i==0 or ((i+1)//2)%2==0) else COLORS['white']
            cx, cy = self.coord_to_pixel(r, c); pygame.draw.circle(self.screen, color, (cx, cy), GRID_SIZE//2-2)
            if color == COLORS['white']: pygame.draw.circle(self.screen, (150,150,150), (cx, cy), GRID_SIZE//2-2, 1)
            if i >= len(moves) - 2: pygame.draw.circle(self.screen, COLORS['highlight'], (cx, cy), 6)

        px = START_X + 756 + 50

        # --- [UI] 状态文字移至顶部 ---
        st = "复盘中" if self.replay_idx != -1 else ("AI 思考..." if self.ai_thinking else ("结束" if self.winner else "进行中"))
        if self.winner and self.replay_idx==-1: st = f"{'黑' if self.winner==Player.BLACK else '白'}方胜"
        status_color = COLORS['highlight'] if self.winner else (COLORS['btn_blue'] if self.ai_thinking else COLORS['text'])
        self.screen.blit(get_font(36, True).render(st, True, status_color), (px, 15))

        pygame.draw.rect(self.screen, (30,30,30), (px, 60, 220, 90), border_radius=8); self.screen.blit(get_font(36, True).render(f"{int(self.time_b)//60:02d}:{int(self.time_b)%60:02d}", True, COLORS['timer_normal']), (px+60, 80)); self.screen.blit(get_font(16).render(f"黑: {self.team_b_name}", True, (200,200,200)), (px+20, 120))
        pygame.draw.rect(self.screen, (220,220,220), (px, 170, 220, 90), border_radius=8); self.screen.blit(get_font(36, True).render(f"{int(self.time_w)//60:02d}:{int(self.time_w)%60:02d}", True, COLORS['timer_normal']), (px+60, 190)); self.screen.blit(get_font(16).render(f"白: {self.team_w_name}", True, (50,50,50)), (px+20, 230))
        self.draw_move_list()

        # [UI] 绘制存档状态提示 (底部)
        if self.save_status_msg:
            msg_color = COLORS['btn_green'] if "✅" in self.save_status_msg else COLORS['btn_red']
            self.screen.blit(get_font(18).render(self.save_status_msg, True, msg_color), (px, 650))

        for b in self.game_ui_elements: b.draw(self.screen)

    def run(self):
        while True:
            if self.state == 'GAME':
                self.update_timer()
                if self.replay_idx == -1: self.check_ai(); self.process_ai()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if self.state == 'MENU':
                    if self.modal.visible:
                        if self.modal.handle_event(event): continue
                    for el in self.menu_ui_elements: el.handle_event(event)
                else:
                    for el in self.game_ui_elements: el.handle_event(event)
                    if event.type == pygame.MOUSEBUTTONDOWN and self.game_active and not self.winner and not self.ai_thinking and self.replay_idx == -1:
                        curr_human = False
                        if self.env.current_player == Player.BLACK and not self.agent_black: curr_human = True
                        elif self.env.current_player == Player.WHITE and not self.agent_white: curr_human = True
                        if curr_human:
                            coord = self.pixel_to_coord(*event.pos)
                            if coord and self.env.is_valid_move(*coord):
                                is_book_move = False
                                found, book_move_coords = self.check_opening_book_moves()
                                if found:
                                    if coord in book_move_coords: is_book_move = True; print(f"✨ [人类] 走出了定式! ({self.get_log_coord(*coord)})")
                                self.env.step(self.env.coord_to_action(*coord))
                                if is_book_move: self.move_sources.append("Book")
                                else: self.move_sources.append("Human")
                                if self.env.game_over: self.finish_game(self.env.winner)
            if self.state == 'MENU': self.draw_menu()
            else: self.draw_game()
            pygame.display.flip(); self.clock.tick(30)

if __name__ == "__main__":
    Connect6GUI().run()
