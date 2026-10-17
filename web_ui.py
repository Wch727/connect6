"""Local browser UI. Run from the repository: python -X utf8 web_ui.py."""
import argparse
import copy
import json
import mimetypes
import pickle
import numpy as np
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from game.connect6_env import Connect6Env, Player
from game.heuristic_ai import HeuristicAI

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'web'


class InvalidRequest(ValueError):
    pass


class GameSession:
    def __init__(self, model_dir=None, ai_factory=None, clock=time.monotonic, opening_book=None):
        self.model_dir = Path(model_dir) if model_dir else ROOT / 'checkpoints'
        self.ai_factory = ai_factory or self._make_ai
        self.book = opening_book
        self.book_message = ''
        if self.book is None:
            path = ROOT / 'connect6_book.pkl'
            try:
                # Trusted local project asset; never accept uploaded pickle files.
                with path.open('rb') as stream:
                    self.book = pickle.load(stream)
                if not isinstance(self.book, dict):
                    raise ValueError('opening book must be a dictionary')
            except FileNotFoundError:
                self.book = {}
                self.book_message = '未找到 connect6_book.pkl，使用 AI 搜索。'
            except Exception as exc:
                self.book = {}
                self.book_message = '开局库读取失败，使用 AI 搜索。'
                print('Opening book load failed:', type(exc).__name__, flush=True)
        self.move_sources = []
        self.clock = clock
        self.lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.generation = 0
        self.env = Connect6Env()
        self.config = {'human': 1, 'model': 'heuristic', 'simulations': 100,
                       'device': 'cpu', 'seconds': 600, 'rules': True, 'book': True}
        self.remaining = {1: 600.0, -1: 600.0}
        self.active = False
        self.thinking = False
        self.error = ''
        self.reason = ''
        self.agent = None
        self.checkpoints = []
        self.stamp = self.clock()

    def models(self):
        return sorted(p.name for p in self.model_dir.glob('*.pt')
                      if p.is_file() and not p.is_symlink())

    def _make_ai(self, config):
        if config['model'] == 'heuristic':
            return HeuristicAI(noise_prob=0.0)
        # Lazy import keeps the heuristic UI usable with NumPy alone.
        from play import Connect6Player
        path = self.model_dir / config['model']
        if not path.is_file() or path.is_symlink():
            raise InvalidRequest('模型文件已不存在，请重新选择。')
        return Connect6Player(model_path=str(path), num_simulations=config['simulations'],
                              device=config['device'], use_rule_based=config['rules'])

    def _tick(self):
        now = self.clock()
        if self.active and not self.env.game_over and not self.error:
            player = int(self.env.current_player)
            self.remaining[player] = max(0.0, self.remaining[player] - (now-self.stamp))
            if self.remaining[player] == 0:
                self.env.game_over = True
                self.env.winner = Player(-player)
                self.reason = 'timeout'
                self.thinking = False
        self.stamp = now

    def state(self):
        with self.lock:
            self._tick()
            return {'board': self.env.board.tolist(), 'player': int(self.env.current_player),
                    'remaining_stones': self.env.get_remaining_moves(),
                    'moves': [{'row': int(r), 'col': int(c),
                               'source': self.move_sources[i],
                               'player': 1 if i == 0 or ((i+1)//2) % 2 == 0 else -1}
                              for i, (r, c) in enumerate(self.env.last_moves)],
                    'winner': int(self.env.winner) if self.env.winner is not None else None,
                    'over': self.env.game_over, 'active': self.active,
                    'thinking': self.thinking, 'error': self.error, 'reason': self.reason,
                    'clock': {'black': self.remaining[1], 'white': self.remaining[-1]},
                    'config': dict(self.config), 'can_undo': bool(self.checkpoints),
                    'book': {'entries': len(self.book), 'message': self.book_message},
                    'generation': self.generation}

    def new(self, data):
        if not isinstance(data, dict):
            raise InvalidRequest('对局设置必须是 JSON 对象。')
        cfg = {k: data.get(k, v) for k, v in self.config.items()}
        if type(cfg['human']) is not int or cfg['human'] not in (1, -1):
            raise InvalidRequest('请选择执黑或执白。')
        if not isinstance(cfg['model'], str) or cfg['model'] not in ['heuristic', *self.models()]:
            raise InvalidRequest('请选择已有的本地模型。')
        if type(cfg['simulations']) is not int or not 1 <= cfg['simulations'] <= 2000:
            raise InvalidRequest('搜索次数应在 1 到 2000 之间。')
        if cfg['device'] not in ('cpu', 'cuda', 'auto'):
            raise InvalidRequest('设备只支持 CPU、CUDA 或自动选择。')
        if type(cfg['seconds']) is not int or not 60 <= cfg['seconds'] <= 7200:
            raise InvalidRequest('每方用时应在 1 到 120 分钟之间。')
        if type(cfg['book']) is not bool:
            raise InvalidRequest('开局库设置无效。')
        if type(cfg['rules']) is not bool:
            raise InvalidRequest('规则辅助设置无效。')
        with self.lock:
            self.generation += 1
            self.env = Connect6Env()
            self.config = cfg
            self.remaining = {1: float(cfg['seconds']), -1: float(cfg['seconds'])}
            self.agent = None
            self.checkpoints = []
            self.move_sources = []
            self.active = True
            self.thinking = False
            self.error = self.reason = ''
            self.stamp = self.clock()
            self._schedule()
            return self.state()

    def move(self, action):
        with self.lock:
            self._tick()
            if not self.active or self.env.game_over or self.error:
                raise InvalidRequest('请先开始一局可进行的对局。')
            if self.thinking or self.env.current_player != self.config['human']:
                raise InvalidRequest('请等待 AI 落子。')
            if type(action) is not int or not 0 <= action < 361 or not self.env.get_valid_moves()[action]:
                raise InvalidRequest('此处不能落子，请选择空交叉点。')
            self.checkpoints.append((copy.deepcopy(self.env), dict(self.remaining)))
            self.env.step(action)
            self.move_sources.append('human')
            if self.env.game_over:
                self.reason = 'line' if self.env.winner else 'draw'
            self._schedule()
            return self.state()

    def undo(self):
        with self.lock:
            if not self.checkpoints:
                raise InvalidRequest('还没有可以撤回的落子。')
            self.generation += 1
            self.env, self.remaining = self.checkpoints.pop()
            self.move_sources = self.move_sources[:len(self.env.last_moves)]
            self.thinking = False
            self.error = self.reason = ''
            self.stamp = self.clock()
            return self.state()

    def _schedule(self):
        if not self.env.game_over and self.env.current_player != self.config['human']:
            self.thinking = True
            generation = self.generation
            threading.Thread(target=self._think, args=(generation,), daemon=True).start()

    def _think(self, generation):
        # Only one model may load/infer at a time. Stale jobs cannot modify a new game.
        with self.inference_lock:
            try:
                with self.lock:
                    if generation != self.generation or self.env.game_over:
                        return
                    cfg, agent = dict(self.config), self.agent
                while True:
                    with self.lock:
                        self._tick()
                        if generation != self.generation or self.env.game_over:
                            return
                        self.agent = agent
                        env = copy.deepcopy(self.env)
                    candidates = self._book_moves(env.board) if cfg['book'] else []
                    source = 'book' if candidates else 'ai'
                    if candidates:
                        row, col = candidates[0]
                        action = row * 19 + col
                    else:
                        if agent is None:
                            agent = self.ai_factory(cfg)
                        action = int(agent.get_action(env))
                    with self.lock:
                        self._tick()
                        if generation != self.generation or self.env.game_over:
                            return
                        if not 0 <= action < 361 or not self.env.get_valid_moves()[action]:
                            raise InvalidRequest('AI 返回了非法落子。')
                        self.env.step(action)
                        self.agent = agent
                        self.move_sources.append(source)
                        if self.env.game_over:
                            self.reason = 'line' if self.env.winner else 'draw'
                            return
                        if self.env.current_player == cfg['human']:
                            return
            except Exception as exc:
                print('AI search failed:', type(exc).__name__, str(exc), flush=True)
                with self.lock:
                    if generation == self.generation:
                        self._tick()
                        self.error = 'AI 未能完成落子。请检查终端提示及模型兼容性，或改用启发式 AI 重新开始。'
            finally:
                with self.lock:
                    if generation == self.generation:
                        self.thinking = False

    def _book_moves(self, board):
        raw = np.where(board == -1, 2, board)
        for turns in range(4):
            rotated = np.rot90(raw, turns)
            for flipped in (False, True):
                transformed = np.fliplr(rotated) if flipped else rotated
                candidates = []
                for row, col in self.book.get(tuple(transformed.flat), []):
                    row, col = int(row), int(col)
                    if not (0 <= row < 19 and 0 <= col < 19):
                        continue
                    if flipped:
                        col = 18-col
                    for _ in range(turns):
                        row, col = col, 18-row
                    if board[row, col] == 0:
                        candidates.append((row, col))
                if candidates:
                    return candidates
        return []

    def export(self):
        with self.lock:
            state = self.state()
            if not state['active']:
                raise InvalidRequest('请先开始对局。')
            return {'format': 'connect6-web-v1', 'board_size': 19,
                    'rules': 'black opens with one stone; two thereafter; six or more wins',
                    'settings': state['config'], 'winner': state['winner'],
                    'finished': state['over'], 'reason': state['reason'],
                    'moves': [{**m, 'coordinate': f"{chr(65+m['col'])}{19-m['row']}"}
                              for m in state['moves']]}


def make_server(session, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, value, kind='application/json; charset=utf-8'):
            payload = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def local_request(self):
            host = self.headers.get('Host', '')
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if host not in allowed:
                self.reply(403, {'error': '只接受本地访问。'})
                return False
            origin = self.headers.get('Origin')
            if origin and origin != f'http://{host}':
                self.reply(403, {'error': '请求来源不匹配。'})
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            path = urlsplit(self.path).path
            if path == '/api/state':
                return self.reply(200, session.state())
            if path == '/api/models':
                return self.reply(200, {'models': session.models()})
            if path == '/api/export':
                try:
                    return self.reply(200, session.export())
                except InvalidRequest as exc:
                    return self.reply(400, {'error': str(exc)})
            assets = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
            if path not in assets:
                return self.reply(404, {'error': '页面不存在。'})
            file = ASSETS / assets[path]
            return self.reply(200, file.read_bytes(), (mimetypes.guess_type(file.name)[0] or 'text/plain')+'; charset=utf-8')

        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8192 or self.headers.get_content_type() != 'application/json':
                    raise InvalidRequest('请求需要有效的 JSON 数据。')
                # Drain bounded POST bodies before rejecting: unread data can reset TCP on Windows.
                body = self.rfile.read(length)
                if not self.local_request():
                    return
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise InvalidRequest('请求需要 JSON 对象。')
                path = urlsplit(self.path).path
                if path == '/api/new':
                    result = session.new(data)
                elif path == '/api/move':
                    result = session.move(data.get('action'))
                elif path == '/api/undo':
                    result = session.undo()
                else:
                    return self.reply(404, {'error': '接口不存在。'})
                self.reply(200, result)
            except (ValueError, TypeError, KeyError) as exc:
                self.reply(400, {'error': str(exc) if isinstance(exc, InvalidRequest) else '请求参数无效。'})
    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description='Local Connect6 browser UI')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--model-dir', type=Path, default=ROOT/'checkpoints')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    server = make_server(GameSession(args.model_dir), args.port)
    print(f'Connect6: http://127.0.0.1:{server.server_port}', flush=True)
    print('Local only. Press Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
