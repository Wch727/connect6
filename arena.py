#!/usr/bin/env python3
"""
竞技场 (Arena): Neural AI vs Heuristic AI (V3.0 Kernel)
------------------------------------------------------------
功能：让神经网络模型与 V3.0 版本的启发式算法自动对战。
用法：
    python arena.py --model checkpoints/iter200.pt --games 10
    python arena.py  (自动搜索最新模型)
"""

import sys
import os
import glob
import numpy as np
import datetime
import argparse
import itertools
import traceback
import time

# --- 路径设置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)

try:
    from game.connect6_env import Connect6Env, Player
    from play import Connect6Player
    from numba import jit
    print("✅ 环境检查通过: Numba 加速已启用")
except ImportError as e:
    print(f"❌ 导入失败: {e}\n请安装 numba 并确保 game/ 和 play.py 存在。"); sys.exit(1)

# ==========================================
# Heuristic V3.0 核心逻辑 (移植版)
# ==========================================
SCORE_WIN       = 100_000_000
SCORE_LIVE_5    = 20_000_000
SCORE_LIVE_4    = 2_000_000
SCORE_DEAD_4    = 500_000
SCORE_LIVE_3    = 400_000
SCORE_JUMP_3    = 380_000
SCORE_DEAD_3    = 5_000
SCORE_LIVE_2    = 2_000

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
    if count == 2 and open_ends == 2: return SCORE_LIVE_2
    return count * 10

@jit(nopython=True, cache=True)
def evaluate_board(board, player):
    total_score = 0
    size = 19
    dr = [0, 1, 1, 1]; dc = [1, 0, 1, -1]
    defense_factor = 2.5 # 激进防御系数

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
    kill_moves, urgent_moves = [], []
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

def get_heuristic_v3_action(env):
    """V3.0 决策入口"""
    # 转换 board: Env(Player=1/-1) -> Numba(1/-1)，直接兼容
    board = env.board
    player = env.current_player
    needed = 1 if env.turn == 0 else 2

    # 1. 开局
    if needed == 1:
        if board[9,9] == 0: return [env.coord_to_action(9,9)]
        cands = get_candidates(board, 2)
        best, bv = cands[0], -float('inf')
        for r, c in cands:
            board[r, c] = player
            v = evaluate_board(board, player)
            board[r, c] = 0
            if v > bv: bv, best = v, (r, c)
        return [env.coord_to_action(*best)]

    # 2. 绝杀
    my_kills, my_threats = scan_threats(board, player)
    my_kills = list(set(my_kills))
    if len(my_kills) >= 2: return [env.coord_to_action(r,c) for r,c in list(my_kills)[:2]]
    if len(my_kills) == 1:
        w = my_kills[0]
        cands = get_candidates(board, 2)
        if w in cands: cands.remove(w)
        return [env.coord_to_action(w[0], w[1]), env.coord_to_action(cands[0][0], cands[0][1]) if cands else 0]

    # 3. 绝防
    opp_kills, opp_threats = scan_threats(board, -player)
    forced_moves = set(opp_kills)
    for t in opp_threats: forced_moves.add(t)
    forced_moves = list(forced_moves)

    # 4. 搜索
    cands = get_candidates(board, 2)
    cand_set = set(cands)
    for fm in forced_moves: cand_set.add(fm)
    cands = list(cand_set)

    top_singles = []
    for r, c in cands:
        board[r, c] = player
        score = evaluate_board(board, player)
        board[r, c] = 0
        if (r, c) in forced_moves: score += 500_000_000
        top_singles.append(((r, c), score))

    top_singles.sort(key=lambda x: x[1], reverse=True)
    top_k = [x[0] for x in top_singles[:15]]

    pairs = list(itertools.combinations(top_k, 2))
    if not pairs and top_k: pairs = [top_k[:2]]

    best_pair = pairs[0] if pairs else []
    best_pair_score = -float('inf')

    for p1, p2 in pairs:
        if opp_kills:
            covered = 0
            if p1 in opp_kills: covered += 1
            if p2 in opp_kills: covered += 1
            if covered == 0: continue # 必须堵必杀

        board[p1] = player
        board[p2] = player
        curr_score = evaluate_board(board, player)
        if opp_kills:
            rem_r, rem_c = scan_threats(board, -player) # 简单scan
            if len(rem_r) > 0: curr_score = -float('inf')
        board[p1] = 0
        board[p2] = 0

        if curr_score > best_pair_score:
            best_pair_score = curr_score
            best_pair = [p1, p2]

    return [env.coord_to_action(best_pair[0][0], best_pair[0][1]),
            env.coord_to_action(best_pair[1][0], best_pair[1][1])]

# --- 存谱逻辑 ---
def save_game_record(env, winner, neural_is_black, game_idx):
    if not env.last_moves: return

    team_b = "Neural" if neural_is_black else "HeuristicV3"
    team_w = "HeuristicV3" if neural_is_black else "Neural"
    res = "Draw"
    if winner == Player.BLACK: res = "B_Win"
    elif winner == Player.WHITE: res = "W_Win"

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"records/Arena_G{game_idx+1}_{team_b}_vs_{team_w}_{res}_{ts}.txt"
    os.makedirs("records", exist_ok=True)

    moves = []
    for i, (r, c) in enumerate(env.last_moves):
        color = "B" if i == 0 else ("W" if ((i+1)//2)%2==1 else "B")
        moves.append(f"{color}({chr(ord('A')+c)},{19-r})")

    content = f"{{[C6][{team_b}][{team_w}][{res}][Arena V3];\n{';'.join(moves)}}}"
    try:
        with open(fname, "w", encoding="gb2312") as f: f.write(content)
    except:
        with open(fname, "w", encoding="utf-8") as f: f.write(content)
    print(f"  -> 棋谱已存: {fname}")

# --- 主循环 ---
def run_arena(model_path, num_games=10, sims=50, device='auto'):
    print("=" * 60)
    print(" 🔥  人机巅峰对决: Neural vs Heuristic V3.0  🔥")
    print("=" * 60)
    print(f"Neural Model: {model_path}")
    print(f"Simulations : {sims}")
    print(f"Device      : {device}")
    print(f"Heuristic   : V3.0 (Grandmaster Logic)")
    print("-" * 60)

    try:
        neural_ai = Connect6Player(model_path=model_path, num_simulations=sims, device=device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    wins = {"Neural": 0, "Heuristic": 0, "Draw": 0}

    for i in range(num_games):
        env = Connect6Env()
        neural_is_black = (i % 2 == 0) # 轮流执黑

        black_name = "Neural" if neural_is_black else "Heuristic"
        white_name = "Heuristic" if neural_is_black else "Neural"
        print(f"\n[第 {i+1}/{num_games} 局] 黑: {black_name}  vs  白: {white_name}")

        start_time = time.time()

        while not env.game_over:
            is_neural = (env.current_player == Player.BLACK and neural_is_black) or \
                        (env.current_player == Player.WHITE and not neural_is_black)

            if is_neural:
                # 神经网络落子 (思考时间取决于 sims 和 GPU)
                try:
                    action = neural_ai.get_action(env)
                    env.step(action)
                except:
                    print("Neural AI 崩溃，判负")
                    env.winner = -env.current_player
                    env.game_active = False
            else:
                # 启发式落子 (极速)
                try:
                    actions = get_heuristic_v3_action(env)
                    for act in actions:
                        if not env.game_over: env.step(act)
                except Exception as e:
                    print(f"Heuristic 错误: {e}")
                    traceback.print_exc()
                    break

        duration = time.time() - start_time
        w = env.winner
        winner_name = "Draw"

        if w == Player.BLACK: winner_name = black_name
        elif w == Player.WHITE: winner_name = white_name

        if winner_name == "Neural": wins["Neural"] += 1
        elif winner_name == "Heuristic": wins["Heuristic"] += 1
        else: wins["Draw"] += 1

        print(f"  🏆 获胜: 【{winner_name}】 (耗时: {duration:.1f}s, 手数: {len(env.last_moves)})")
        save_game_record(env, w, neural_is_black, i)

    print("\n" + "="*30)
    print(f"🏁 最终比分 (共{num_games}局)")
    print(f"🧠 Neural:    {wins['Neural']}")
    print(f"⚡ Heuristic: {wins['Heuristic']}")
    print(f"🤝 平局:      {wins['Draw']}")
    print("="*30)

# --- 参数处理 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arena: Neural vs Heuristic")
    parser.add_argument('--model', type=str, default=None, help='Path to .pt model file')
    parser.add_argument('--games', type=int, default=10, help='Number of games')
    parser.add_argument('--sims', type=int, default=50, help='Neural MCTS simulations')
    parser.add_argument('--device', type=str, default='auto', help='cpu or cuda')

    args = parser.parse_args()

    # 自动搜寻模型
    final_path = args.model
    if not final_path:
        candidates = glob.glob("checkpoints/*.pt") + glob.glob("*.pt")
        if candidates:
            # 按修改时间排序，取最新的
            candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            final_path = candidates[0]
            print(f"🔍 自动选择模型: {final_path}")
        else:
            print("❌ 未找到 .pt 模型文件，无法启动 Neural AI。")
            sys.exit(1)

    run_arena(final_path, args.games, args.sims, args.device)
