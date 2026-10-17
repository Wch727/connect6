#!/usr/bin/env python3
"""
智奕连城 - 模型竞技场 (Benchmark)
---------------------------------------------------------
功能:
1. 加载 checkpoints 文件夹下的所有模型。
2. 让模型两两进行对弈 (不显示界面，纯后台计算)。
3. 统计胜率并输出报表。
"""

import os
import sys
import glob
import torch
import time
import numpy as np
from tqdm import tqdm  # 进度条库，如果没有请 pip install tqdm

# --- 路径修正 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

try:
    from game.connect6_env import Connect6Env, Player
    from play import Connect6Player
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ================= 配置区域 =================
# 模型文件夹路径
MODEL_DIR = os.path.join(CURRENT_DIR, "checkpoints")

# 比赛局数 (每对模型，交换先后手各下多少局)
# 例如设为 10，则 A执黑10局，A执白10局，共 20 局
GAMES_PER_SIDE = 5

# AI 思考强度 (模拟次数)
# 评测时建议稍微降低一点以节省时间，或者设高一点以求精准
SIMULATIONS = 200

# 设备 (自动检测)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ===========================================

def load_all_models():
    """扫描并加载所有模型"""
    patterns = [os.path.join(MODEL_DIR, "*.pt"), os.path.join(CURRENT_DIR, "*.pt")]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))

    # 去重并排序
    paths = sorted(list(set(paths)))

    if not paths:
        print("❌ 未找到任何模型文件 (.pt)！")
        return []

    print(f"🔍 找到 {len(paths)} 个模型文件，准备加载...")

    agents = []
    for path in paths:
        name = os.path.basename(path)
        try:
            # 初始化 AI
            agent = Connect6Player(model_path=path, num_simulations=SIMULATIONS, device=DEVICE)
            agents.append({"name": name, "agent": agent, "path": path})
            print(f"  ✅ 已加载: {name}")
        except Exception as e:
            print(f"  ❌ 加载失败 {name}: {e}")

    return agents

def play_one_game(agent_b, agent_w):
    """
    进行一局游戏
    :return: 1 (黑胜), -1 (白胜), 0 (平局)
    """
    env = Connect6Env()
    env.reset()

    while not env.game_over:
        if env.current_player == Player.BLACK:
            # 黑方行动
            try:
                action = agent_b.get_action(env)
                env.step(action)
            except:
                return -1 # 黑方崩盘判负
        else:
            # 白方行动
            try:
                action = agent_w.get_action(env)
                env.step(action)
            except:
                return 1 # 白方崩盘判负

    if env.winner == Player.BLACK: return 1
    elif env.winner == Player.WHITE: return -1
    return 0

def run_matchup(model_1, model_2):
    """
    两个模型之间的对抗 (交换先后手)
    """
    name1 = model_1['name']
    name2 = model_2['name']

    stats = {
        "p1_black_wins": 0, # model1 执黑胜
        "p1_white_wins": 0, # model1 执白胜
        "draws": 0,
        "total": 0
    }

    print(f"\n⚔️  正在对决: {name1} VS {name2}")

    # 进度条
    total_games = GAMES_PER_SIDE * 2
    pbar = tqdm(total=total_games, ncols=80, desc="对弈进度")

    # Round 1: Model 1 执黑
    for _ in range(GAMES_PER_SIDE):
        res = play_one_game(model_1['agent'], model_2['agent'])
        if res == 1: stats["p1_black_wins"] += 1
        elif res == 0: stats["draws"] += 1
        stats["total"] += 1
        pbar.update(1)

    # Round 2: Model 1 执白 (Model 2 执黑)
    for _ in range(GAMES_PER_SIDE):
        res = play_one_game(model_2['agent'], model_1['agent'])
        # res=1是黑胜(Model2), res=-1是白胜(Model1)
        if res == -1: stats["p1_white_wins"] += 1
        elif res == 0: stats["draws"] += 1
        stats["total"] += 1
        pbar.update(1)

    pbar.close()
    return stats

def print_report(results):
    print("\n" + "="*70)
    print(f"{'📊 胜率排行榜':^60}")
    print("="*70)
    print(f"{'模型名称':<25} | {'总胜率':<8} | {'执黑胜率':<8} | {'执白胜率':<8} | {'净胜场':<6}")
    print("-" * 70)

    # 按总胜率排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['win_rate'], reverse=True)

    for name, data in sorted_results:
        print(f"{name:<25} | {data['win_rate']:>7.1%} | {data['black_rate']:>7.1%} | {data['white_rate']:>7.1%} | {data['net_wins']:>6}")
    print("="*70)

def main():
    print("🚀 智奕连城 AI 竞技场启动...")
    print(f"⚙️  设置: 每对对弈 {GAMES_PER_SIDE*2} 局 | 思考模拟 {SIMULATIONS} 次 | 设备 {DEVICE}")

    models = load_all_models()
    if len(models) < 2:
        print("⚠️ 至少需要 2 个模型才能进行对战！")
        return

    # 全局计分板
    scoreboard = {}
    for m in models:
        scoreboard[m['name']] = {
            "wins": 0, "games": 0, "black_wins": 0, "black_games": 0, "white_wins": 0, "white_games": 0
        }

    # 循环赛逻辑 (Round Robin)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1 = models[i]
            m2 = models[j]

            # 跑比赛
            match_res = run_matchup(m1, m2)

            # 统计 Model 1 数据
            s1 = scoreboard[m1['name']]
            s1['black_wins'] += match_res['p1_black_wins']
            s1['black_games'] += GAMES_PER_SIDE
            s1['white_wins'] += match_res['p1_white_wins']
            s1['white_games'] += GAMES_PER_SIDE
            s1['wins'] += (match_res['p1_black_wins'] + match_res['p1_white_wins'])
            s1['games'] += match_res['total']

            # 统计 Model 2 数据 (反推)
            s2 = scoreboard[m2['name']]
            # M2 执黑胜 = 总局数 - 平局 - M1执白胜
            m2_black_wins = GAMES_PER_SIDE - (match_res['total'] - GAMES_PER_SIDE * 2) - match_res['p1_white_wins'] # 简化逻辑：暂且认为无平局或简单计算
            # 更精确的方式：
            # M1 执黑胜 -> M2 执白负
            # M1 执白胜 -> M2 执黑负

            # 重新精确统计 M2
            # M2 执黑局 (即Round 2)
            # Round 2 总局数 GAMES_PER_SIDE. M1胜(白) = p1_white_wins.
            # 该历史统计将未胜局全部计为另一方胜局。
            m2_black_win_count = GAMES_PER_SIDE - match_res['p1_white_wins'] # 此处未单列和棋。
            m2_white_win_count = GAMES_PER_SIDE - match_res['p1_black_wins']

            s2['black_wins'] += m2_black_win_count
            s2['black_games'] += GAMES_PER_SIDE
            s2['white_wins'] += m2_white_win_count
            s2['white_games'] += GAMES_PER_SIDE
            s2['wins'] += (m2_black_win_count + m2_white_win_count)
            s2['games'] += match_res['total']

    # 计算最终百分比
    final_stats = {}
    for name, data in scoreboard.items():
        total_g = data['games'] if data['games'] > 0 else 1
        b_g = data['black_games'] if data['black_games'] > 0 else 1
        w_g = data['white_games'] if data['white_games'] > 0 else 1

        final_stats[name] = {
            "win_rate": data['wins'] / total_g,
            "black_rate": data['black_wins'] / b_g,
            "white_rate": data['white_wins'] / w_g,
            "net_wins": data['wins'] - (total_g - data['wins']) # 净胜
        }

    print_report(final_stats)

if __name__ == "__main__":
    main()
