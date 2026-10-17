import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm
import sys
import gc # 垃圾回收

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.connect6_env import Connect6Env
from game.heuristic_ai import HeuristicAI
from training.trainer import augment_data

def generate_dataset(total_games=3000, batch_size=1000, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"🚀 计划生成 {total_games} 局数据，每 {batch_size} 局保存一次...")

    ai = HeuristicAI(noise_prob=0.1)
    env = Connect6Env()

    # 当前批次的数据
    batch_data = []

    for i in tqdm(range(total_games), desc="生成中"):
        env.reset()
        game_record = []
        while not env.game_over:
            state = env.get_state()
            action = ai.get_action(env)
            policy = np.zeros(361, dtype=np.float32)
            policy[action] = 1.0
            game_record.append((state, policy, env.current_player))
            env.step(action)

        winner = env.winner
        for state, policy, player in game_record:
            if winner is None: value = 0.0
            else: value = 1.0 if player == winner else -1.0
            batch_data.extend(augment_data(state, policy, value))

        # === 关键修改：每满 batch_size 局，保存一次并释放内存 ===
        if (i + 1) % batch_size == 0:
            part_idx = (i + 1) // batch_size
            save_path = os.path.join(output_dir, f"sl_data_part_{part_idx}.pkl")

            print(f"\n💾 正在保存第 {part_idx} 部分 ({len(batch_data)} 样本)...")
            with open(save_path, 'wb') as f:
                pickle.dump(batch_data, f)

            print(f"✅ 已保存至 {save_path}")

            # 清空列表，强制释放内存
            batch_data = []
            gc.collect()

    # 保存剩余不足一个 batch 的数据
    if len(batch_data) > 0:
        save_path = os.path.join(output_dir, f"sl_data_final.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(batch_data, f)
        print("✅ 保存剩余数据完成")

if __name__ == "__main__":
    generate_dataset(total_games=3000, batch_size=500)
