"""Local Katagomo GTP connection. No engine or model is redistributed."""
import copy
import os
from pathlib import Path
import queue
import re
import subprocess
import tempfile
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
COLS='ABCDEFGHJKLMNOPQRST'


def to_gtp(action):
    if type(action) is not int or not 0<=action<361:raise ValueError('落子坐标无效。')
    return COLS[action%19]+str(19-action//19)


def from_gtp(value):
    match=re.fullmatch(r'([A-HJ-T])([1-9]|1[0-9])',value.strip().upper())
    if not match:raise ValueError('Katagomo 没有返回有效落子：'+value)
    return (19-int(match[2]))*19+COLS.index(match[1])


def find_package(explicit=None):
    candidates=[]
    if explicit:candidates.append(Path(explicit))
    elif os.environ.get('KATAGOMO_DIR'):candidates.append(Path(os.environ['KATAGOMO_DIR']))
    else:
        for parent in (ROOT.parent,ROOT.parent.parent):
            candidates.extend(sorted(parent.glob('Katagomo*')))
    for path in candidates:
        if all((path/name).is_file() for name in ('engine/connectsix19x_opencl.exe','weights/connectsix19x_b18trans.bin.gz','engine.cfg')):
            return path.resolve()
    raise FileNotFoundError('未找到 Katagomo。请用 --katagomo-dir 指定软件文件夹。')


class GTP:
    def __init__(self,package,seconds):
        run_dir=ROOT/'runs'/'katagomo_ui';run_dir.mkdir(parents=True,exist_ok=True)
        self.directory=tempfile.TemporaryDirectory(prefix='engine-',dir=run_dir)
        folder=Path(self.directory.name)
        raw=(package/'engine.cfg').read_bytes()
        try:config=raw.decode('utf-8-sig')
        except UnicodeDecodeError:config=raw.decode('gb18030')
        values={'numSearchThreads':8,'nnCacheSizePowerOfTwo':18,'nnMutexPoolSizePowerOfTwo':16,
                'ponderingEnabled':'false','lagBuffer':0,'maxTime':seconds,'allowResignation':'false',
                'logDir':(folder/'logs').as_posix(),'logToStderr':'false'}
        for key,value in values.items():
            config=re.sub(r'^\s*'+re.escape(key)+r'\s*=.*$', '',config,flags=re.MULTILINE)
            config+=f'\n{key} = {value}\n'
        cfg=folder/'engine.cfg';cfg.write_text(config,encoding='utf-8')
        self.stderr=(folder/'stderr.log').open('w',encoding='utf-8')
        self.proc=None;self.lock=threading.RLock();self.lines=queue.Queue()
        try:
            self.proc=subprocess.Popen([str(package/'engine/connectsix19x_opencl.exe'),'gtp','-config',str(cfg),
                                        '-model',str(package/'weights/connectsix19x_b18trans.bin.gz')],
                cwd=package,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,
                encoding='utf-8',errors='replace',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            def read():
                try:
                    for line in self.proc.stdout:self.lines.put(line)
                finally:self.lines.put(None)
            self.reader=threading.Thread(target=read,daemon=True);self.reader.start()
            self.command('name',120)
        except Exception:
            self.close();raise

    def command(self,command,timeout=30):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:raise RuntimeError('Katagomo 引擎已退出，请重新开始。')
            try:
                self.proc.stdin.write(command+'\n');self.proc.stdin.flush()
                end=time.monotonic()+timeout;reply=[];started=False
                while True:
                    line=self.lines.get(timeout=max(.001,end-time.monotonic()))
                    if line is None:raise RuntimeError('Katagomo 引擎异常退出，请检查显卡驱动。')
                    if not started:
                        if line.startswith(('=','?')):started=True;reply.append(line.rstrip())
                    elif not line.strip():break
                    else:reply.append(line.rstrip())
                value='\n'.join(reply)
                if value.startswith('?'):raise ValueError('Katagomo 拒绝命令：'+value[1:].strip())
                return value[1:].strip()
            except (queue.Empty,BrokenPipeError,OSError) as exc:
                self.close()
                raise RuntimeError('Katagomo 响应超时或连接断开，请重新开始。') from exc

    def close(self):
        with self.lock:
            if self.proc is not None:
                if self.proc.poll() is None:self.proc.kill()
                self.proc.wait(timeout=5)
                if hasattr(self,'reader'):self.reader.join(2)
                self.proc.stdin.close();self.proc.stdout.close();self.proc=None
            self.stderr.close()
            self.directory.cleanup()


class Katagomo:
    def __init__(self,package_dir=None,seconds=1,transport_factory=None):
        if type(seconds) not in (int,float) or not .1<=seconds<=30:raise ValueError('思考时间应在 0.1 到 30 秒之间。')
        self.transport=None;self.moves=[];self.closed=False
        try:
            self.transport=transport_factory() if transport_factory else GTP(find_package(package_dir),seconds)
            self.transport.command('boardsize 19');self.transport.command('clear_board');self.transport.command('komi 0')
            self.transport.command(f'time_settings 0 {max(1,int(seconds+0.999))} 1')
        except Exception:
            self.close();raise

    @staticmethod
    def history(env):return [int(r)*19+int(c) for r,c in env.last_moves]

    def sync(self,env):
        wanted=self.history(env)
        if self.moves!=wanted[:len(self.moves)]:
            self.transport.command('clear_board');self.moves=[]
        for i in range(len(self.moves),len(wanted)):
            player='b' if i==0 or ((i+1)//2)%2==0 else 'w'
            self.transport.command(f'play {player} {to_gtp(wanted[i])}');self.moves.append(wanted[i])

    def get_action(self,env):
        if self.closed:raise RuntimeError('Katagomo 引擎已关闭。')
        self.sync(env)
        result=self.transport.command('genmove '+('b' if env.current_player==1 else 'w'),45)
        action=from_gtp(result)
        if not env.get_valid_moves()[action]:raise ValueError('Katagomo 返回了已占用的位置。')
        self.moves.append(action)
        return action

    def recommend(self,env):
        if env.game_over:return []
        self.sync(env);original=len(self.moves);trial=copy.deepcopy(env);player=trial.current_player;hints=[]
        try:
            while not trial.game_over and trial.current_player==player:
                action=self.get_action(trial);hints.append(action);trial.step(action)
            return hints
        finally:
            while len(self.moves)>original:
                self.transport.command('undo');self.moves.pop()

    def close(self):
        if not self.closed and self.transport is not None:self.transport.close()
        self.closed=True
