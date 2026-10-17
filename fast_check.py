#!/usr/bin/env python3
"""
智奕连城 - 模型竞技场 (V2.0 显眼包版)
---------------------------------------------------------
特点:
1. 实时打印每一步，防止你以为它死机了。
2. 限制最大步数，防止陷入僵局。
3. 降低模拟次数，快速出结果。
"""

import os
import sys
import glob
import torch
import time
import numpy as np

# --- 路径修正 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

try:
    from game.connect6_env import Connect6Env, Player
    from play import Connect6Player
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ================= ⚡ 速测配置 ⚡ =================
MODEL_DIR = os.path.join(CURRENT_DIR, "checkpoints")

# 每对模型对战局数 (A执黑1局，A执白1局)
GAMES_PER_SIDE = 1

# 模拟次数 (调低这个数字！CPU建议 20-50，GPU建议 200-400)
# 之前的 200 对 CPU 来说太慢了
SIMULATIONS = 50

# 强制结束步数 (防止长考拖时间，超过这个步数判平局)
MAX_MOVES = 150

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ==================================================

def load_all_models():
    patterns = [os.path.join(MODEL_DIR, "*.pt"), os.path.join(CURRENT_DIR, "*.pt")]
    paths = []
    for p in patterns: paths.extend(glob.glob(p))
    paths = sorted(list(set(paths)))

    if not paths:
        print("❌ 未找到任何模型文件！请检查 checkpoints 文件夹。")
        return []

    print(f"🔍 发现 {len(paths)} 个模型:")
    agents = []
    for path in paths:
        name = os.path.basename(path)
        try:
            # 只有这里需要加载一次，比较慢
            print(f"  ⏳ 正在加载 {name} ... ", end="", flush=True)
            agent = Connect6Player(model_path=path, num_simulations=SIMULATIONS, device=DEVICE)
            agents.append({"name": name, "agent": agent})
            print("✅ 完成")
        except Exception as e:
            print(f"❌ 失败 ({e})")

    return agents

def play_one_game(agent_b, agent_w, names):
    """
    进行一局游戏 (带实时输出)
    """
    env = Connect6Env()
    env.reset()

    print(f"\n   🏁 开始: 黑[{names[0]}] vs 白[{names[1]}]")

    # 限制最大步数，防止死循环
    for move_count in range(1, MAX_MOVES + 1):
        start_time = time.time()

        if env.current_player == Player.BLACK:
            player_name = "黑"
            action = agent_b.get_action(env)
        else:
            player_name = "白"
            action = agent_w.get_action(env)

        env.step(action)
        duration = time.time() - start_time

        # 实时打印落子信息
        print(f"      Step {move_count:02d}: {player_name}方落子... (耗时 {duration:.1f}s)")

        if env.game_over:
            if env.winner == Player.BLACK:
                print(f"      🎉 结果: 黑方[{names[0]}] 胜！")
                return 1
            elif env.winner == Player.WHITE:
                print(f"      🎉 结果: 白方[{names[1]}] 胜！")
                return -1
            else:
                print(f"      🤝 结果: 平局")
                return 0

    print(f"      ⚠️ 超过 {MAX_MOVES} 步未分胜负，强制平局。")
    return 0

def main():
    print(f"🚀 启动快速评测 (设备: {DEVICE})")
    print("-" * 50)

    models = load_all_models()
    if len(models) < 2:
        print("⚠️ 至少需要 2 个模型才能对战！请复制几个 .pt 文件过来。")
        return

    # 计分板
    scores = {m['name']: 0 for m in models}
    matches = {m['name']: 0 for m in models}

    # 循环赛
    total_pairs = len(models) * (len(models) - 1) // 2
    pair_idx = 0

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            pair_idx += 1
            m1 = models[i]
            m2 = models[j]

            print("\n" + "="*60)
            print(f"⚔️  第 {pair_idx}/{total_pairs} 组对决: {m1['name']} VS {m2['name']}")
            print("="*60)

            # Round 1: M1 黑
            res = play_one_game(m1['agent'], m2['agent'], (m1['name'], m2['name']))
            if res == 1: scores[m1['name']] += 1
            elif res == -1: scores[m2['name']] += 1
            # else 平局不加分

            matches[m1['name']] += 1
            matches[m2['name']] += 1

            # Round 2: M2 黑 (交换)
            res = play_one_game(m2['agent'], m1['agent'], (m2['name'], m1['name']))
            if res == 1: scores[m2['name']] += 1
            elif res == -1: scores[m1['name']] += 1

            matches[m1['name']] += 1
            matches[m2['name']] += 1

    print("\n\n" + "#"*60)
    print(f"{'🏆 最终战报 🏆':^56}")
    print("#"*60)
    print(f"{'模型名称':<25} | {'胜场':<6} | {'总场次':<6} | {'胜率':<6}")
    print("-" * 60)

    # 排序
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for name, score in sorted_scores:
        total = matches[name]
        rate = (score / total * 100) if total > 0 else 0
        print(f"{name:<25} | {score:<6} | {total:<6} | {rate:.1f}%")
    print("#"*60)

if __name__ == "__main__":
    main()
