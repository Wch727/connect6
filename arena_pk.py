#!/usr/bin/env python3
"""
双模型竞技场 (Neural PK Arena)
---------------------------------------------------------
功能：加载两个不同的神经网络模型，让它们互相对战。
机制：每局结束后自动换先（轮流执黑），确保公平性。
输出：实时战况 + 自动保存棋谱 + 最终胜率统计。
"""

import sys
import os
import argparse
import time
import glob
from datetime import datetime
import numpy as np

# --- 路径修正 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from game.connect6_env import Connect6Env, Player
    from play import Connect6Player
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保 game/ 和 play.py 在当前目录下。")
    sys.exit(1)

# --- 工具函数 ---
def get_model_name(path):
    """从路径中提取简短的模型名"""
    if not path: return "Random"
    return os.path.splitext(os.path.basename(path))[0]

def save_pk_record(env, winner, p1_is_black, name1, name2, game_idx):
    """保存PK对局棋谱"""
    if not env.last_moves: return

    # 确定当前局谁是黑方
    black_name = name1 if p1_is_black else name2
    white_name = name2 if p1_is_black else name1

    res_str = "Draw"
    if winner == Player.BLACK: res_str = "B_Win"
    elif winner == Player.WHITE: res_str = "W_Win"

    # 文件名：Arena_局数_黑方名_vs_白方名_结果.txt
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"records/PK_G{game_idx+1}_{black_name}_vs_{white_name}_{res_str}_{ts}.txt"
    os.makedirs("records", exist_ok=True)

    # 构造棋谱内容
    moves_list = []
    for i, (r, c) in enumerate(env.last_moves):
        # 0->B, 1,2->W ...
        color = "B" if i == 0 else ("W" if ((i+1)//2)%2==1 else "B")
        coord = f"{chr(ord('A') + c)},{19 - r}"
        moves_list.append(f"{color}({coord})")

    full_content = (
        f"{{[C6][{black_name}][{white_name}][{res_str}]"
        f"[{datetime.now().strftime('%Y.%m.%d %H:%M')} PK];\n"
        f"{';'.join(moves_list)}}}"
    )

    try:
        with open(filename, "w", encoding="gb2312") as f: f.write(full_content)
    except:
        with open(filename, "w", encoding="utf-8") as f: f.write(full_content)
    print(f"    └── 棋谱已保存: {filename}")

# --- 竞技场主逻辑 ---
def run_pk(model1_path, model2_path, num_games, simulations, device):
    print("=" * 60)
    print(" 🔥  六子棋巅峰对决 (Neural PK)  🔥")
    print("=" * 60)

    name1 = get_model_name(model1_path)
    name2 = get_model_name(model2_path)

    print(f"选手 1: {name1} (路径: {model1_path})")
    print(f"选手 2: {name2} (路径: {model2_path})")
    print(f"对局数: {num_games}")
    print(f"思考量: {simulations} sims/move")
    print(f"设  备: {device}")
    print("-" * 60)

    # 1. 加载模型
    try:
        print("正在加载模型 1...", end="", flush=True)
        player1 = Connect6Player(model_path=model1_path, num_simulations=simulations, device=device)
        print(" ✅")

        print("正在加载模型 2...", end="", flush=True)
        player2 = Connect6Player(model_path=model2_path, num_simulations=simulations, device=device)
        print(" ✅")
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        return

    # 统计数据
    stats = {name1: 0, name2: 0, "Draw": 0}

    # 2. 开始对战循环
    for i in range(num_games):
        env = Connect6Env()

        # 轮流执黑：偶数局(0,2...) P1执黑，奇数局(1,3...) P2执黑
        p1_is_black = (i % 2 == 0)

        current_black = name1 if p1_is_black else name2
        current_white = name2 if p1_is_black else name1

        print(f"\n[第 {i+1}/{num_games} 局] 黑方: {current_black}  vs  白方: {current_white}")
        start_time = time.time()

        # 对局循环
        while not env.game_over:
            # 判断当前是谁的回合
            is_p1_turn = False
            if env.current_player == Player.BLACK:
                if p1_is_black: is_p1_turn = True
            else:
                if not p1_is_black: is_p1_turn = True

            # 对应的玩家落子
            player_to_move = player1 if is_p1_turn else player2
            action = player_to_move.get_action(env, deterministic=False) # PK时建议带一点随机性防止完全重复
            env.step(action)

            # 可选：打印步数进度
            # print(f"\r  步数: {len(env.last_moves)}", end="")

        duration = time.time() - start_time

        # 3. 结算
        winner_name = "Draw"
        if env.winner == Player.BLACK:
            winner_name = current_black
        elif env.winner == Player.WHITE:
            winner_name = current_white

        stats[winner_name] += 1

        print(f"  🏆 本局获胜: 【{winner_name}】 (耗时: {duration:.1f}s, 手数: {len(env.last_moves)})")

        # 保存棋谱
        save_pk_record(env, env.winner, p1_is_black, name1, name2, i)

    # 4. 最终统计
    print("\n" + "=" * 60)
    print("📊 最终战报")
    print("=" * 60)
    print(f"选手 1 ({name1}): 胜 {stats[name1]} 局 ({(stats[name1]/num_games)*100:.1f}%)")
    print(f"选手 2 ({name2}): 胜 {stats[name2]} 局 ({(stats[name2]/num_games)*100:.1f}%)")
    print(f"平  局: {stats['Draw']} 局")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network PK Arena")
    parser.add_argument('--model1', type=str, required=True, help='路径到模型1 (.pt)')
    parser.add_argument('--model2', type=str, required=True, help='路径到模型2 (.pt)')
    parser.add_argument('--games', type=int, default=10, help='对战局数')
    parser.add_argument('--sims', type=int, default=50, help='MCTS模拟次数 (建议 50-200)')
    parser.add_argument('--device', type=str, default='auto', help='运行设备 (cpu/cuda)')

    args = parser.parse_args()

    run_pk(args.model1, args.model2, args.games, args.sims, args.device)
