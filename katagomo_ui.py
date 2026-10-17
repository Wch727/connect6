"""Simple local Katagomo board: python katagomo_ui.py (NumPy required)."""
import argparse
import copy
import json
import math
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from game.connect6_env import Connect6Env
from game.katagomo import Katagomo, find_package, to_gtp
from game.records import parse_record, record_env, export_sgf

ROOT=Path(__file__).resolve().parent


def player_at(index):return 1 if index==0 or ((index+1)//2)%2==0 else -1


class Session:
    def __init__(self,package_dir=None,engine_factory=None):
        self.package_dir=package_dir;self.factory=engine_factory or (lambda seconds:Katagomo(package_dir,seconds))
        self.available=bool(engine_factory)
        if not engine_factory:
            try:find_package(package_dir);self.available=True
            except FileNotFoundError:pass
        self.lock=threading.RLock();self.engine_lock=threading.Lock();self.agent=None;self.agent_seconds=None
        self.env=Connect6Env();self.actions=[];self.human=1;self.seconds=1.;self.mode='play'
        self.active=False;self.busy=False;self.job='';self.error='';self.hints=[];self.version=0;self.closed=False

    def state(self):
        with self.lock:
            cursor=len(self.env.last_moves)
            return dict(board=self.env.board.tolist(),moves=[dict(action=a,row=a//19,col=a%19,player=player_at(i),coordinate=to_gtp(a)) for i,a in enumerate(self.actions)],
                cursor=cursor,total=len(self.actions),player=int(self.env.current_player),remaining_stones=self.env.get_remaining_moves(),
                over=self.env.game_over,winner=int(self.env.winner) if self.env.winner is not None else None,
                human=self.human,seconds=self.seconds,mode=self.mode,active=self.active,busy=self.busy,job=self.job,error=self.error,
                hints=list(self.hints),version=self.version,engine_available=self.available,
                can_undo=bool(cursor and (self.mode=='review' or any(player_at(i)==self.human for i in range(cursor)))))

    def _invalidate(self):
        self.version+=1;self.busy=False;self.job='';self.hints=[];self.error=''

    def _settings(self,data):
        if not isinstance(data,dict):raise ValueError('设置必须是 JSON 对象。')
        human=data.get('human',self.human);seconds=data.get('seconds',self.seconds);mode=data.get('mode',self.mode)
        if type(human) is not int or human not in (1,-1):raise ValueError('请选择执黑或执白。')
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or not .1<=seconds<=30:
            raise ValueError('每颗思考时间应在 0.1 到 30 秒之间。')
        if mode not in ('play','review'):raise ValueError('请选择对弈或复盘。')
        return human,float(seconds),mode

    def new(self,data):
        with self.lock:
            human,seconds,mode=self._settings(data)
            self._invalidate();self.human,self.seconds,self.mode=human,seconds,mode
            self.env=Connect6Env();self.actions=[];self.active=True
            self._schedule();return self.state()

    def configure(self,data):
        with self.lock:
            if self.busy:raise ValueError('请在本次思考结束后调整时间。')
            _,seconds,_=self._settings(data);self.seconds=seconds
            return self.state()

    def _schedule(self):
        if self.mode=='play' and not self.env.game_over and self.env.current_player!=self.human:self._start('play')

    def move(self,action):
        with self.lock:
            if not self.active or self.env.game_over:raise ValueError('请开始新局，或回退到未结束的局面。')
            if self.busy or self.error:raise ValueError('请等待思考结束，或重新开始。')
            if self.mode=='play' and self.env.current_player!=self.human:raise ValueError('请等待 Katagomo 落子。')
            if type(action) is not int or not 0<=action<361 or not self.env.get_valid_moves()[action]:raise ValueError('请选择空交叉点。')
            self.actions=self.actions[:len(self.env.last_moves)];self._invalidate()
            self.env.step(action);self.actions.append(action);self._schedule();return self.state()

    def seek(self,index):
        with self.lock:
            if type(index) is not int or not 0<=index<=len(self.actions):raise ValueError('复盘位置无效。')
            env=record_env(self.actions[:index]);self._invalidate();self.env=env;self.mode='review';self.active=True
            return self.state()

    def undo(self):
        with self.lock:
            cursor=len(self.env.last_moves)
            if self.mode=='review':
                if not cursor:raise ValueError('已经到棋谱开头。')
                return self.seek(cursor-1)
            indices=[i for i in range(cursor) if player_at(i)==self.human]
            if not indices:raise ValueError('你还没有落子。')
            self._invalidate();self.actions=self.actions[:indices[-1]];self.env=record_env(self.actions)
            return self.state()

    def load(self,data):
        if not isinstance(data,dict):raise ValueError('棋谱数据无效。')
        actions=parse_record(data.get('text'));env=record_env(actions)
        with self.lock:
            self._invalidate();self.actions=actions;self.env=env;self.active=True;self.mode='review'
            return self.state()

    def resume(self,data):
        with self.lock:
            human,seconds,_=self._settings(data)
            if self.env.game_over:raise ValueError('请先回退到尚未结束的局面。')
            self._invalidate();self.human=human;self.seconds=seconds;self.mode='play';self.active=True
            self.actions=self.actions[:len(self.env.last_moves)];self._schedule();return self.state()

    def analyze(self):
        with self.lock:
            if not self.active or self.env.game_over:raise ValueError('请选择尚未结束的局面。')
            if self.busy:raise ValueError('正在思考，请稍候。')
            self._invalidate();self._start('hint');return self.state()

    def apply_hint(self):
        with self.lock:
            if self.busy or not self.hints:raise ValueError('请先获取当前局面的推荐。')
            if self.mode=='play' and self.env.current_player!=self.human:raise ValueError('请等待 Katagomo 落子。')
            hints=list(self.hints);self.actions=self.actions[:len(self.env.last_moves)];self._invalidate()
            for a in hints:
                if self.env.game_over or not self.env.get_valid_moves()[a]:raise ValueError('推荐已失效，请重新分析。')
                self.env.step(a);self.actions.append(a)
            self._schedule();return self.state()

    def export(self,sgf=False):
        with self.lock:
            if sgf:return {'sgf':export_sgf(self.actions)}
            return dict(format='connect6-web-v1',board_size=19,
                        moves=[dict(action=a,row=a//19,col=a%19,player=player_at(i),coordinate=to_gtp(a)) for i,a in enumerate(self.actions)])

    def _start(self,job):
        self.busy=True;self.job=job;version=self.version
        threading.Thread(target=self._run,args=(version,job),daemon=True).start()

    def _run(self,version,job):
        with self.engine_lock:
            try:
                with self.lock:
                    if self.closed or self.version!=version:return
                    seconds=self.seconds
                if self.agent is None or self.agent_seconds!=seconds:
                    if self.agent is not None:self.agent.close();self.agent=None
                    self.agent=self.factory(seconds);self.agent_seconds=seconds
                while True:
                    with self.lock:
                        if self.closed or self.version!=version:return
                        env=copy.deepcopy(self.env)
                    if job=='hint':
                        hints=self.agent.recommend(env)
                        with self.lock:
                            if self.version==version:self.hints=list(hints)
                        return
                    action=int(self.agent.get_action(env))
                    with self.lock:
                        if self.closed or self.version!=version:return
                        if not 0<=action<361 or not self.env.get_valid_moves()[action]:raise ValueError('Katagomo 返回了非法落子。')
                        self.env.step(action);self.actions.append(action)
                        if self.env.game_over or self.env.current_player==self.human:return
            except Exception as exc:
                print('Katagomo:',type(exc).__name__,str(exc),flush=True)
                if self.agent is not None:self.agent.close();self.agent=None
                with self.lock:
                    if self.version==version:self.error=str(exc)
            finally:
                with self.lock:
                    if self.version==version:self.busy=False;self.job=''

    def close(self):
        with self.lock:self.closed=True;self._invalidate()
        with self.engine_lock:
            if self.agent is not None:self.agent.close();self.agent=None


def make_server(session,port=8766):
    assets={'/':('katagomo.html','text/html'),'/katagomo.js':('katagomo.js','text/javascript'),
            '/style.css':('style.css','text/css'),'/katagomo.css':('katagomo.css','text/css'),'/favicon.svg':('katagomo-icon.svg','image/svg+xml')}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,code,data,kind='application/json'):
            body=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False).encode('utf-8')
            self.send_response(code);self.send_header('Content-Type',kind+'; charset=utf-8');self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)
        def local(self):
            host=self.headers.get('Host','');allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            if host not in allowed or self.headers.get('Origin',f'http://{host}')!=f'http://{host}':
                self.reply(403,{'error':'只接受同源的本地访问。'});return False
            return True
        def do_GET(self):
            if not self.local():return
            path=urlsplit(self.path).path
            if path=='/api/state':return self.reply(200,session.state())
            if path in ('/api/export','/api/export-sgf'):return self.reply(200,session.export(path.endswith('-sgf')))
            if path in assets:
                name,kind=assets[path];return self.reply(200,(ROOT/'web'/name).read_bytes(),kind)
            self.reply(404,{'error':'页面不存在。'})
        def do_POST(self):
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=1048576 or self.headers.get_content_type()!='application/json':raise ValueError('请求需要有效 JSON。')
                # Consume bounded bodies before a 403 to avoid Windows connection resets.
                body=self.rfile.read(length)
                if not self.local():return
                data=json.loads(body)
                if not isinstance(data,dict):raise ValueError('请求需要 JSON 对象。')
                routes={'/api/new':lambda:session.new(data),'/api/move':lambda:session.move(data.get('action')),
                    '/api/undo':session.undo,'/api/seek':lambda:session.seek(data.get('index')),
                    '/api/import':lambda:session.load(data),'/api/resume':lambda:session.resume(data),
                    '/api/settings':lambda:session.configure(data),'/api/analyze':session.analyze,'/api/apply-hint':session.apply_hint}
                call=routes.get(urlsplit(self.path).path)
                if call is None:return self.reply(404,{'error':'接口不存在。'})
                self.reply(200,call())
            except (ValueError,TypeError,KeyError) as exc:self.reply(400,{'error':str(exc)})
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main():
    parser=argparse.ArgumentParser(description='Katagomo 简洁对弈与复盘界面')
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--katagomo-dir',type=Path)
    args=parser.parse_args()
    if not 1<=args.port<=65535:parser.error('port must be between 1 and 65535')
    session=Session(args.katagomo_dir);server=make_server(session,args.port)
    print(f'Katagomo: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();session.close()


if __name__=='__main__':main()
