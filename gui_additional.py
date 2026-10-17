#!/usr/bin/env python3
"""
智奕连城 - 人机混合对抗赛 (V9.3 完整UI修复版)
---------------------------------------------------------
[修复内容]
1. UI修复: 恢复棋盘左侧(1-19)和底部(A-S)的坐标刻度。
2. 功能修复: 比赛结束后自动显示复盘按钮，支持单步回放，棋谱列表同步更新。
3. 核心逻辑: 保持 1-2-2-2 连动规则 (黑1人 -> 白1,2机 -> 黑2,3机 -> 白3,4人...)。
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

# --- 基础配置 ---
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from game.connect6_env import Connect6Env, Player
    from play import Connect6Player
except ImportError as e:
    print(f"❌ 错误: 缺少核心文件 (game/connect6_env.py 或 play.py)")
    sys.exit(1)

# --- 界面参数 ---
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
GRID_SIZE = 42
BOARD_SIZE = 19
START_X = 80
START_Y = 60
TIME_LIMIT = 600
BOOK_ACCEPT_RATE = 0.95

COLORS = {
    'bg': (240, 242, 245), 'board': (220, 179, 92), 'grid': (0, 0, 0),
    'black': (15, 15, 15), 'white': (245, 245, 245), 'highlight': (231, 76, 60),
    'text': (44, 62, 80), 'sub_text': (100, 100, 100),
    'timer_normal': (46, 204, 113), 'btn_blue': (52, 152, 219),
    'btn_red': (231, 76, 60), 'btn_gray': (149, 165, 166), 'btn_green': (39, 174, 96),
    'input_bg': (255, 255, 255), 'input_border': (189, 195, 199), 'input_active': (52, 152, 219),
    'panel_bg': (255, 255, 255), 'modal_overlay': (0, 0, 0, 150), 'modal_bg': (255, 255, 255),
    'log_text': (60, 60, 60),
    'tag_book': (39, 174, 96), 'tag_ai': (52, 152, 219), 'tag_human': (155, 89, 182)
}

def get_font(size, bold=False):
    font_path = os.path.join(BASE_DIR, "font.ttf")
    if os.path.exists(font_path):
        try: return pygame.font.Font(font_path, size)
        except: pass
    try: return pygame.font.SysFont("simhei", size, bold=bold)
    except: return pygame.font.Font(None, size)

# --- 核心逻辑类 ---
class SymmetryHandler:
    @staticmethod
    def get_canonical_key(board_np):
        flat = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                v = board_np[r][c]
                flat.append(1 if v==1 else (2 if v==-1 else 0))
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
class Button:
    def __init__(self, x, y, w, h, text, func, color_theme='btn_blue', font_size=24):
        self.rect = pygame.Rect(x, y, w, h); self.text = text; self.func = func
        self.base_color = COLORS.get(color_theme, COLORS['btn_blue']); self.hover = False; self.visible = True
        self.font = get_font(font_size)
    def draw(self, screen):
        if not self.visible: return
        c = tuple(min(v+20, 255) for v in self.base_color) if self.hover else self.base_color
        pygame.draw.rect(screen, (180,180,180), (self.rect.x, self.rect.y+3, self.rect.w, self.rect.h), border_radius=6)
        pygame.draw.rect(screen, c, self.rect, border_radius=6)
        txt = self.font.render(self.text, True, (255,255,255))
        screen.blit(txt, txt.get_rect(center=self.rect.center))
    def handle_event(self, event):
        if not self.visible: return
        if event.type == pygame.MOUSEMOTION: self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and self.hover: self.func()

class InputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h); self.color = COLORS['input_border']; self.text = text; self.active = False; self.font = get_font(24)
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos): self.active = True; self.color = COLORS['input_active']; pygame.key.start_text_input()
            else: self.active = False; self.color = COLORS['input_border']
        if self.active:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN: self.active = False; self.color = COLORS['input_border']
                elif event.key == pygame.K_BACKSPACE: self.text = self.text[:-1]
            elif event.type == pygame.TEXTINPUT: self.text += event.text
    def draw(self, screen):
        pygame.draw.rect(screen, COLORS['input_bg'], self.rect, border_radius=5)
        pygame.draw.rect(screen, self.color, self.rect, 3 if self.active else 1, border_radius=5)
        txt = self.font.render(self.text if self.text else "点击输入...", True, COLORS['text'] if self.text else COLORS['sub_text'])
        screen.blit(txt, (self.rect.x+10, self.rect.centery-txt.get_height()//2))

class ModelSelectButton(Button):
    def __init__(self, x, y, w, h, current_model, open_modal_func):
        super().__init__(x, y, w, h, "", open_modal_func, 'input_bg'); self.current_model_path = current_model; self.font = get_font(20)
    def update_model(self, path): self.current_model_path = path
    def draw(self, screen):
        pygame.draw.rect(screen, COLORS['input_bg'], self.rect, border_radius=5)
        pygame.draw.rect(screen, COLORS['input_active'] if self.hover else COLORS['input_border'], self.rect, 2, border_radius=5)
        name = "不使用模型 (手动)" if not self.current_model_path else os.path.basename(self.current_model_path)
        txt = self.font.render(name, True, COLORS['text'])
        screen.blit(txt, (self.rect.x+10, self.rect.centery-txt.get_height()//2))

class ModelListModal:
    def __init__(self, models, callback_func):
        self.models = [("不使用模型 (手动)", None)] + models; self.callback = callback_func; self.visible = False
        self.w, self.h = 600, 500; self.x = (WINDOW_WIDTH-self.w)//2; self.y = (WINDOW_HEIGHT-self.h)//2
        self.rect = pygame.Rect(self.x, self.y, self.w, self.h); self.scroll_y = 0; self.item_h = 50
        self.close_btn = Button(self.x+self.w-40, self.y+10, 30, 30, "X", self.hide, 'btn_red', 18)
    def show(self): self.visible = True; self.scroll_y = 0
    def hide(self): self.visible = False
    def handle_event(self, event):
        if not self.visible: return False
        self.close_btn.handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4: self.scroll_y = min(0, self.scroll_y + 30)
            elif event.button == 5: self.scroll_y = max(-(len(self.models)*self.item_h - 440), self.scroll_y - 30)
            elif event.button == 1:
                if pygame.Rect(self.x, self.y+50, self.w, 440).collidepoint(event.pos):
                    idx = (event.pos[1] - (self.y+50) - self.scroll_y) // self.item_h
                    if 0 <= idx < len(self.models): self.callback(self.models[idx][1]); self.hide()
        return True
    def draw(self, screen):
        if not self.visible: return
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA); overlay.fill(COLORS['modal_overlay']); screen.blit(overlay, (0,0))
        pygame.draw.rect(screen, COLORS['modal_bg'], self.rect, border_radius=10); pygame.draw.rect(screen, COLORS['btn_blue'], self.rect, 2, border_radius=10)
        self.close_btn.draw(screen)
        list_s = pygame.Surface((self.w-20, 440)); list_s.fill(COLORS['modal_bg'])
        for i, (name, _) in enumerate(self.models):
            y = i * self.item_h + self.scroll_y
            if -self.item_h < y < 440:
                pygame.draw.rect(list_s, (245,245,245), (0, y, self.w-20, self.item_h-5), border_radius=5)
                list_s.blit(get_font(20).render(name, True, COLORS['text']), (10, y+12))
        screen.blit(list_s, (self.x+10, self.y+50))

# --- 主程序 ---
class Connect6GUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("智奕连城 - 混合对抗赛系统 (V9.3)")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_mono = get_font(16)

        # 加载开局库
        self.opening_book = {}
        book_path = os.path.join(BASE_DIR, "connect6_book.pkl")
        if os.path.exists(book_path):
            try:
                with open(book_path, "rb") as f: self.opening_book = pickle.load(f)
                print(f"✅ 开局库已加载: {len(self.opening_book)} 条目")
            except: pass

        self.available_models = self.scan_models()
        self.state = 'MENU'
        self.menu_ui = self.init_menu_ui()
        self.game_ui = []
        self.modal = ModelListModal(self.available_models, self.on_model_selected)

        self.env = None
        self.agent_black = None; self.agent_white = None
        self.path_b = None; self.path_w = None
        self.team_b_name = "Team1"; self.team_w_name = "Team2"

        self.game_active = False; self.winner = None
        self.ai_thinking = False; self.ai_action_queue = None
        self.move_sources = []
        self.has_saved = False; self.save_status_msg = ""
        self.time_b = TIME_LIMIT; self.time_w = TIME_LIMIT; self.last_time = 0; self.replay_idx = -1

    def scan_models(self):
        paths = glob.glob(os.path.join(BASE_DIR, "*.pt")) + glob.glob(os.path.join(BASE_DIR, "checkpoints", "*.pt"))
        return [(os.path.basename(p), os.path.abspath(p)) for p in list(set(paths))]

    def init_menu_ui(self):
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        return [
            InputBox(cx-350, cy-60, 250, 45, "Team_1"),
            ModelSelectButton(cx-350, cy+10, 250, 45, None, lambda: self.open_modal('black')),
            InputBox(cx+100, cy-60, 250, 45, "Team_2"),
            ModelSelectButton(cx+100, cy+10, 250, 45, None, lambda: self.open_modal('white')),
            Button(cx-100, cy+150, 200, 60, "开始比赛", self.start_game, 'btn_green', 28)
        ]

    def open_modal(self, role): self.current_selecting_role = role; self.modal.show()
    def on_model_selected(self, path):
        if self.current_selecting_role == 'black': self.path_b = path; self.menu_ui[1].update_model(path)
        else: self.path_w = path; self.menu_ui[3].update_model(path)

    def start_game(self):
        self.team_b_name = self.menu_ui[0].text; self.team_w_name = self.menu_ui[2].text
        self.agent_black = self.load_agent(self.path_b); self.agent_white = self.load_agent(self.path_w)
        self.env = Connect6Env(); self.reset_game_logic()
        self.init_game_ui()
        self.state = 'GAME'

    def init_game_ui(self):
        px, py = START_X + 756 + 50, 520
        self.game_ui = [
            Button(px, py, 140, 50, "悔棋", self.undo_move, 'btn_gray'),
            Button(px+160, py, 140, 50, "重置", self.reset_game, 'btn_red'),
            Button(px, py+70, 140, 50, "< 上一步", self.replay_prev, 'btn_blue'),
            Button(px+160, py+70, 140, 50, "下一步 >", self.replay_next, 'btn_blue'),
            Button(px+80, py+140, 140, 50, "菜单", self.back_to_menu, 'btn_gray')
        ]
        # 复盘按钮初始隐藏
        self.game_ui[2].visible = False; self.game_ui[3].visible = False

    def load_agent(self, path):
        if not path: return None
        try:
            import torch
            return Connect6Player(model_path=path, num_simulations=800, device='cuda' if torch.cuda.is_available() else 'cpu')
        except: return None

    def reset_game(self): self.env.reset(); self.reset_game_logic()
    def reset_game_logic(self):
        self.game_active = True; self.winner = None; self.ai_thinking = False; self.ai_action_queue = None
        self.move_sources = []; self.has_saved = False; self.save_status_msg = ""
        self.time_b = TIME_LIMIT; self.time_w = TIME_LIMIT; self.last_time = time.time(); self.replay_idx = -1
        # 重置时隐藏复盘按钮
        if self.game_ui:
            self.game_ui[2].visible = False; self.game_ui[3].visible = False

    def back_to_menu(self): self.state = 'MENU'

    def get_current_actor_type(self):
        """混合赛规则: 1-2-2-2 (黑1人 -> 白1,2机 -> 黑2,3机 -> 白3,4人...)"""
        n = len(self.env.last_moves)
        turn_id = 0 if n == 0 else (n + 1) // 2
        team_turn_count = turn_id // 2
        if self.env.current_player == Player.BLACK:
            return "HUMAN" if team_turn_count % 2 == 0 else "AI"
        else:
            return "AI" if team_turn_count % 2 == 0 else "HUMAN"

    def check_opening_book_moves(self):
        if not self.opening_book: return False, []
        symmetries = SymmetryHandler.get_symmetries(np.array(self.env.board))
        for transformed_board, k, is_flipped in symmetries:
            key = SymmetryHandler.get_canonical_key(transformed_board)
            if key in self.opening_book:
                moves = self.opening_book[key]
                if moves:
                    real = [SymmetryHandler.inverse_coord(r, c, k, is_flipped) for r, c in moves]
                    real = [(r, c) for r, c in real if self.env.is_valid_move(r, c)]
                    if real:
                        return True, real
        return False, []

    def check_ai(self):
        if not self.game_active or self.winner or self.ai_thinking or self.replay_idx != -1: return
        if self.get_current_actor_type() == "AI":
            agent = self.agent_black if self.env.current_player == Player.BLACK else self.agent_white
            if agent:
                self.ai_thinking = True
                threading.Thread(target=self.ai_worker, args=(agent,), daemon=True).start()

    def ai_worker(self, agent):
        try:
            found, move_coords_list = self.check_opening_book_moves()
            if found and random.random() < BOOK_ACCEPT_RATE:
                self.ai_action_queue = self.env.coord_to_action(*move_coords_list[0])
                self.move_sources.append("Book")
                time.sleep(0.5); return

            action = agent.get_action(self.env)
            self.ai_action_queue = action
            self.move_sources.append("AI")
        except: self.ai_thinking = False

    def process_ai(self):
        if self.ai_thinking and self.ai_action_queue is not None:
            self.env.step(self.ai_action_queue)
            self.ai_thinking = False; self.ai_action_queue = None
            if self.env.game_over: self.finish_game(self.env.winner)

    def handle_human_click(self, pos):
        if not self.game_active or self.winner or self.ai_thinking or self.replay_idx != -1: return
        if self.get_current_actor_type() == "HUMAN":
            coord = self.pixel_to_coord(*pos)
            if coord and self.env.is_valid_move(*coord):
                source_tag = "Human"
                found, book_moves = self.check_opening_book_moves()
                if found and coord in book_moves: source_tag = "Book"

                self.env.step(self.env.coord_to_action(*coord))
                self.move_sources.append(source_tag)
                if self.env.game_over: self.finish_game(self.env.winner)

    def undo_move(self):
        if self.ai_thinking or not self.game_active or not self.env.last_moves: return
        moves = self.env.last_moves[:-1]
        self.env.reset(); self.move_sources = self.move_sources[:len(moves)]
        for r, c in moves: self.env.step(self.env.coord_to_action(r, c))

    def replay_prev(self):
        if self.replay_idx == -1: self.replay_idx = len(self.env.last_moves)
        if self.replay_idx > 0: self.replay_idx -= 1
    def replay_next(self):
        if self.replay_idx != -1 and self.replay_idx < len(self.env.last_moves): self.replay_idx += 1

    def finish_game(self, winner):
        self.winner = winner; self.game_active = False; self.auto_save_log()
        # 显示复盘按钮
        self.game_ui[2].visible = True; self.game_ui[3].visible = True

    def auto_save_log(self):
        if self.has_saved: return
        res = "先手胜" if self.winner == Player.BLACK else ("后手胜" if self.winner == Player.WHITE else "和棋")
        filename = f"C6-{self.team_b_name}_vs_{self.team_w_name}-{res}.txt"
        os.makedirs(os.path.join(BASE_DIR, "records"), exist_ok=True)
        try:
            with open(os.path.join(BASE_DIR, "records", filename), "w", encoding="gbk") as f:
                f.write(f"{{[C6][{self.team_b_name}][{self.team_w_name}][{res}][{datetime.now()}]}}")
            self.save_status_msg = f"✅ 棋谱已保存: {filename}"
        except: self.save_status_msg = "❌ 存档失败"
        self.has_saved = True

    def pixel_to_coord(self, px, py):
        c, r = round((px-START_X)/GRID_SIZE), round((py-START_Y)/GRID_SIZE)
        if 0<=r<BOARD_SIZE and 0<=c<BOARD_SIZE: return r, c
        return None
    def get_log_coord(self, r, c): return f"{chr(ord('A') + c)}{19 - r}"

    # --- 绘制逻辑 ---
    def draw_move_list(self):
        px, py = START_X + 756 + 50, 280; w, h = 300, 260
        pygame.draw.rect(self.screen, COLORS['panel_bg'], (px, py, w, h), border_radius=8)
        pygame.draw.rect(self.screen, (200,200,200), (px, py, w, h), 2, border_radius=8)
        self.screen.blit(get_font(18, True).render("实时棋谱 / Move List", True, COLORS['text']), (px+10, py+5))

        moves = self.env.last_moves if self.replay_idx == -1 else self.env.last_moves[:self.replay_idx]
        display_count = 11
        start_idx = max(0, len(moves) - display_count)

        for i in range(start_idx, len(moves)):
            r, c = moves[i]
            tag, tag_col = "", COLORS['sub_text']
            if i < len(self.move_sources):
                src = self.move_sources[i]
                if src == "Book": tag = "[定式]"; tag_col = COLORS['tag_book']
                elif src == "AI": tag = "[AI]"; tag_col = COLORS['tag_ai']
                elif src == "Human": tag = "[人类]"; tag_col = COLORS['tag_human']

            txt = f"{i+1:02d}. {'B' if (i==0 or ((i+1)//2)%2==0) else 'W'}({self.get_log_coord(r,c)})"
            y = py + 30 + (i - start_idx) * 20
            self.screen.blit(self.font_mono.render(txt, True, COLORS['log_text']), (px+20, y))
            self.screen.blit(self.font_mono.render(tag, True, tag_col), (px+160, y))

    def draw_game(self):
        self.screen.fill(COLORS['bg'])
        # 棋盘背景
        pygame.draw.rect(self.screen, COLORS['board'], (START_X-20, START_Y-20, 756+40, 756+40), border_radius=5)

        # 绘制网格和坐标 (恢复标度!)
        for i in range(19):
            # 网格线
            pygame.draw.line(self.screen, COLORS['grid'], (START_X, START_Y+i*GRID_SIZE), (START_X+756, START_Y+i*GRID_SIZE))
            pygame.draw.line(self.screen, COLORS['grid'], (START_X+i*GRID_SIZE, START_Y), (START_X+i*GRID_SIZE, START_Y+756))
            # 左侧数字 19-1
            label_num = get_font(16).render(str(19-i), True, COLORS['text'])
            self.screen.blit(label_num, (START_X-30, START_Y+i*GRID_SIZE-8))
            # 底部字母 A-S
            label_char = get_font(16).render(chr(ord('A')+i), True, COLORS['text'])
            self.screen.blit(label_char, (START_X+i*GRID_SIZE-5, START_Y+756+10))

        # 星位
        for r in [3,9,15]:
            for c in [3,9,15]:
                pygame.draw.circle(self.screen, COLORS['grid'], (START_X+c*GRID_SIZE, START_Y+r*GRID_SIZE), 4)

        # 棋子
        moves = self.env.last_moves if self.replay_idx == -1 else self.env.last_moves[:self.replay_idx]
        for i, (r, c) in enumerate(moves):
            color = COLORS['black'] if (i==0 or ((i+1)//2)%2==0) else COLORS['white']
            cx, cy = START_X+c*GRID_SIZE, START_Y+r*GRID_SIZE
            pygame.draw.circle(self.screen, color, (cx, cy), 19)
            if color == COLORS['white']: pygame.draw.circle(self.screen, (0,0,0), (cx, cy), 19, 1)
            # 最后一手高亮
            if i == len(moves)-1: pygame.draw.circle(self.screen, COLORS['highlight'], (cx, cy), 6)
            # 偶数手连子标记
            if i>0 and len(moves)%2!=1 and i==len(moves)-1:
                pr, pc = moves[i-1]; pygame.draw.circle(self.screen, COLORS['highlight'], (START_X+pc*GRID_SIZE, START_Y+pr*GRID_SIZE), 4)

        px = START_X + 756 + 50
        # 状态
        actor = self.get_current_actor_type() if not self.winner else "OVER"
        st_txt = f"[{'人类' if actor=='HUMAN' else 'AI机器人'}] 操作" if not self.winner else "比赛结束"
        if self.replay_idx != -1: st_txt = "【复盘模式】"
        self.screen.blit(get_font(36, True).render(st_txt, True, COLORS['tag_human'] if actor=='HUMAN' else COLORS['tag_ai']), (px, 15))

        # 计时器
        pygame.draw.rect(self.screen, (30,30,30), (px, 60, 260, 90), border_radius=8)
        self.screen.blit(get_font(36).render(f"{int(self.time_b)//60:02d}:{int(self.time_b)%60:02d}", True, COLORS['timer_normal']), (px+70, 80))
        self.screen.blit(get_font(16).render(f"黑(1队): {self.team_b_name}", True, (200,200,200)), (px+20, 125))

        pygame.draw.rect(self.screen, (220,220,220), (px, 170, 260, 90), border_radius=8)
        self.screen.blit(get_font(36).render(f"{int(self.time_w)//60:02d}:{int(self.time_w)%60:02d}", True, (0,0,0)), (px+70, 190))
        self.screen.blit(get_font(16).render(f"白(2队): {self.team_w_name}", True, (50,50,50)), (px+20, 235))

        self.draw_move_list()
        if self.save_status_msg: self.screen.blit(get_font(18).render(self.save_status_msg, True, COLORS['btn_green']), (px, 680))
        for b in self.game_ui: b.draw(self.screen)

    def run(self):
        while True:
            if self.state == 'MENU':
                self.screen.fill(COLORS['bg'])
                title = get_font(48, True).render("智奕连城 - 混合对抗赛", True, COLORS['text'])
                self.screen.blit(title, (WINDOW_WIDTH//2-title.get_width()//2, 100))
                for el in self.menu_ui: el.draw(self.screen);
                if self.modal.visible: self.modal.draw(self.screen)
            else:
                self.draw_game(); self.check_ai(); self.process_ai()
                now = time.time(); dt = now - self.last_time; self.last_time = now
                if self.game_active and not self.winner:
                    if self.env.current_player == Player.BLACK: self.time_b -= dt
                    else: self.time_w -= dt
                    if self.time_b <= 0 or self.time_w <= 0: self.finish_game(Player.WHITE if self.time_b<=0 else Player.BLACK)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if self.state == 'MENU':
                    if not self.modal.handle_event(event):
                        for el in self.menu_ui: el.handle_event(event)
                else:
                    for el in self.game_ui: el.handle_event(event)
                    if event.type == pygame.MOUSEBUTTONDOWN and self.game_active and not self.winner and not self.ai_thinking and self.replay_idx == -1:
                        if self.get_current_actor_type() == "HUMAN": self.handle_human_click(event.pos)
            pygame.display.flip(); self.clock.tick(30)

if __name__ == "__main__":
    Connect6GUI().run()
