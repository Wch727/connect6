"""Read legal 19x19 Connect6 main lines from SGF or the browser JSON format."""
import json
from game.connect6_env import Connect6Env


def record_env(actions):
    if len(actions)>361:
        raise ValueError('棋谱超过 361 颗。')
    env=Connect6Env()
    for action in actions:
        if type(action) is not int or not 0<=action<361 or env.game_over or not env.get_valid_moves()[action]:
            raise ValueError('棋谱包含重复落子、非法位置或终局之后的落子。')
        env.step(action)
    return env


def _sgf_nodes(text):
    index=0
    def whitespace():
        nonlocal index
        while index<len(text) and text[index].isspace():index+=1
    def tree(depth=0):
        nonlocal index
        if depth>64:raise ValueError('棋谱分支层数过多。')
        whitespace()
        if index>=len(text) or text[index]!='(':raise ValueError('SGF 格式无效。')
        index+=1;nodes=[];first_child=None
        while True:
            whitespace()
            if index>=len(text):raise ValueError('SGF 棋谱未结束。')
            char=text[index]
            if char==')':
                index+=1
                return nodes+(first_child or [])
            if char=='(':
                child=tree(depth+1)
                if first_child is None:first_child=child
                continue
            if char!=';' or first_child is not None:raise ValueError('SGF 节点格式无效。')
            index+=1;node={}
            while True:
                whitespace()
                if index>=len(text) or text[index] in ';()':break
                start=index
                while index<len(text) and text[index].isalpha():index+=1
                key=text[start:index].upper()
                if not key:raise ValueError('SGF 属性格式无效。')
                whitespace();values=[]
                while index<len(text) and text[index]=='[':
                    index+=1;value=[]
                    while True:
                        if index>=len(text):raise ValueError('SGF 属性未结束。')
                        char=text[index];index+=1
                        if char==']':break
                        if char=='\\':
                            if index>=len(text):raise ValueError('SGF 转义无效。')
                            char=text[index];index+=1
                            if char=='\r':
                                if index<len(text) and text[index]=='\n':index+=1
                                continue
                            if char=='\n':continue
                        value.append(char)
                    values.append(''.join(value));whitespace()
                if not values:raise ValueError('SGF 属性缺少值。')
                node.setdefault(key,[]).extend(values)
            nodes.append(node)
    nodes=tree();whitespace()
    if index!=len(text):raise ValueError('一次请导入一份 SGF 棋谱。')
    return nodes


def parse_record(text):
    if not isinstance(text,str) or len(text)>524288:raise ValueError('棋谱文件过大或格式无效。')
    text=text.lstrip('\ufeff').strip()
    if not text:raise ValueError('棋谱文件为空。')
    actions=[];players=[]
    if text.startswith('{'):
        data=json.loads(text)
        if not isinstance(data,dict) or data.get('board_size',19)!=19 or not isinstance(data.get('moves'),list):
            raise ValueError('请选择 19×19 六子棋 JSON 棋谱。')
        for move in data['moves']:
            if not isinstance(move,dict):raise ValueError('棋谱落子格式无效。')
            if 'action' in move:action=move['action']
            else:
                r,c=move.get('row'),move.get('col')
                if type(r) is not int or type(c) is not int or not 0<=r<19 or not 0<=c<19:
                    raise ValueError('棋谱落子坐标无效。')
                action=r*19+c
            actions.append(action);players.append(move.get('player'))
    else:
        for node in _sgf_nodes(text):
            if 'SZ' in node and node['SZ']!=['19']:raise ValueError('只支持 19×19 棋盘。')
            if any(key in node for key in ('AB','AW','AE')):raise ValueError('请导入从空棋盘开始的对局棋谱。')
            if 'B' in node and 'W' in node:raise ValueError('同一个 SGF 节点不能同时落黑白子。')
            for key in ('B','W'):
                for coordinate in node.get(key,[]):
                    if len(coordinate)!=2 or any(ch not in 'abcdefghijklmnopqrs' for ch in coordinate):
                        raise ValueError('棋谱包含停着或超出棋盘的落子。')
                    actions.append((ord(coordinate[1])-97)*19+ord(coordinate[0])-97)
                    players.append(1 if key=='B' else -1)
    if not actions:raise ValueError('棋谱没有落子。')
    record_env(actions)
    for i,player in enumerate(players):
        expected=1 if i==0 or ((i+1)//2)%2==0 else -1
        if player is not None and (type(player) is not int or player!=expected):
            raise ValueError('棋谱黑白顺序不符合六子棋每回合两子的规则。')
    return actions


def export_sgf(actions):
    env=record_env(actions)
    result='B+6' if env.winner==1 else 'W+6' if env.winner==-1 else ''
    text=f'(;FF[4]GM[40]CA[UTF-8]SZ[19]RU[Connect6]AP[Connect6]RE[{result}]'
    for i,a in enumerate(actions):
        color='B' if i==0 or ((i+1)//2)%2==0 else 'W'
        text+=f';{color}[{chr(97+a%19)}{chr(97+a//19)}]'
    return text+')'
