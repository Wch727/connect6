"""Parse KataGomo opening analysis: xy alternatives and white-perspective wl."""
import ast
from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
from game.connect6_env import Connect6Env, Player


@dataclass(frozen=True)
class OpeningPosition:
    board: tuple
    candidates: tuple
    player: Player
    value: float
    visits: float


def _array(content, name):
    match = re.search(r'\bconst\s+' + name + r'\s*=\s*(\[.*?\])\s*;', content, re.S)
    if not match:
        raise ValueError('Missing array: ' + name)
    # Parse literals only: HTML/JavaScript is never executed.
    return ast.literal_eval(match.group(1))


def parse_opening(content, min_value=None):
    """Return the most-visited legal candidate group; values use player-to-act."""
    try:
        board = _array(content, 'board')
        if len(board) != 361 or any(type(x) is not int or x not in (0, 1, 2) for x in board):
            return None
        match = re.search(r'\bconst\s+nextPla\s*=\s*([12])\s*;', content)
        if not match:
            return None
        player = Player.BLACK if match.group(1) == '1' else Player.WHITE
        moves = _array(content, 'moves')
        if not isinstance(moves, list):
            return None
        groups = []
        for move in moves:
            if not isinstance(move, dict) or not isinstance(move.get('xy'), list):
                continue
            wl, visits = move.get('wl'), move.get('v')
            if type(wl) not in (int, float) or type(visits) not in (int, float):
                continue
            if not math.isfinite(wl) or not -1 <= wl <= 1 or not math.isfinite(visits) or visits <= 0:
                continue
            value = -float(wl) if player == Player.BLACK else float(wl)
            if min_value is not None and value < min_value:
                continue
            candidates = []
            for xy in move['xy']:
                if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                    continue
                x, y = xy
                if type(x) is not int or type(y) is not int or not (0 <= x < 19 and 0 <= y < 19):
                    continue
                point = (y, x)
                if board[y*19+x] == 0 and point not in candidates:
                    candidates.append(point)
            if candidates:
                groups.append(OpeningPosition(tuple(board), tuple(candidates), player, value, visits))
        return max(groups, key=lambda group: group.visits) if groups else None
    except (ValueError, TypeError, SyntaxError, OverflowError):
        return None


def read_opening(path, min_value=None):
    try:
        return parse_opening(Path(path).read_text(encoding='utf-8'), min_value)
    except (OSError, UnicodeError):
        return None


def make_sample(position):
    env = Connect6Env()
    board = np.array(position.board, dtype=np.int8).reshape(19, 19)
    env.board = np.where(board == 2, -1, board).astype(np.int8)
    env.current_player = position.player
    # Static analysis pages contain no ordered history. Preserve the existing
    # repeated-position feature convention used by opening-data checkpoints.
    env.history = [env.board.copy() for _ in range(7)]
    stones = int(np.count_nonzero(board))
    env.turn = (stones+1)//2 if stones else 0
    env.move_count = (stones-1) % 2 if stones else 0
    policy = np.zeros(361, dtype=np.float32)
    for row, col in position.candidates:
        policy[row*19+col] = 1.0/len(position.candidates)
    return env.get_state().astype(np.int8), policy, position.value


def augment_sample(sample):
    state, policy, value = sample
    policy = policy.reshape(19, 19)
    for k in range(4):
        rotated_state = np.rot90(state, k, axes=(1, 2)).copy()
        rotated_policy = np.rot90(policy, k).copy()
        yield rotated_state, rotated_policy.flatten(), value
        yield np.flip(rotated_state, axis=2).copy(), np.fliplr(rotated_policy).flatten().copy(), value


def html_files(roots):
    import os
    for root in roots:
        for directory, dirs, files in os.walk(root):
            dirs.sort()
            for name in sorted(files):
                if name.lower().endswith('.html'):
                    yield Path(directory)/name


def prepare_outputs(paths, overwrite=False):
    paths = [Path(p) for p in paths]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('Output paths must be distinct')
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f'{path.name} already exists; choose another output or use --overwrite')
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
