import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from game.connect6_env import Connect6Env, Player
from model.network import get_network
import os
import pickle as _pickle # 导入 pickle 以捕获特定的 UnpicklingError

# --- 配置 ---
MODEL_PATH = "checkpoints/final_model.pt"
BOARD_SIZE = 19
OUTPUT_DIR = "figs"

def plot_heatmap():
    # --- 准备工作：确保输出目录存在 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 初始化环境并摆一个局
    env = Connect6Env()
    env.reset()

    moves = [
        (9,9),          # 黑
        (9,10), (9,11), # 白
        (8,9), (10,9)   # 黑
    ]
    for r, c in moves:
        env.step(r * BOARD_SIZE + c)

    # 2. 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_network('hybrid').to(device)

    print(f"正在从 '{MODEL_PATH}' 加载模型...")

    # --- 最终解决方案 ---
    # 根据新的错误信息，我们需要将 numpy.dtype 也添加到安全全局变量列表中。
    # 这个列表现在包含了所有在模型文件中发现的非标准对象类型。
    torch.serialization.add_safe_globals([
        np.bool_,
        np.uint8,
        np._core.multiarray.scalar,
        np.dtype  # <--- 新增的类型
    ])

    try:
        # 优先使用默认的安全方式加载 (weights_only=True)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 模型加载成功！ (安全模式)")
    except _pickle.UnpicklingError:
        print("⚠️ 安全加载失败，尝试使用 'weights_only=False' ... (请确保模型来源可信)")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 使用 'weights_only=False' 成功加载模型！")
    except FileNotFoundError:
        print(f"❌ 错误：找不到模型文件 '{MODEL_PATH}'。请检查路径是否正确。")
        return
    except Exception as e:
        print(f"❌ 加载模型时发生未知错误: {e}")
        return

    model.eval()

    # 3. 预测
    state = env.get_state()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        policy, value = model(state_tensor)
        probs = torch.softmax(policy, dim=1).cpu().numpy().reshape(BOARD_SIZE, BOARD_SIZE)

    # 4. 画图
    plt.figure(figsize=(12, 10))
    sns.heatmap(probs, cmap="viridis", annot=True, fmt=".2%", linewidths=.5)
    plt.title(f"Policy Heatmap (Value Est: {value.item():.3f})")
    plt.xlabel("Column")
    plt.ylabel("Row")

    current_board = env.get_board().reshape(BOARD_SIZE, BOARD_SIZE)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if current_board[r, c] == Player.BLACK.value:
                plt.text(c + 0.5, r + 0.5, 'B', color='white', ha='center', va='center', fontsize=10, weight='bold')
            elif current_board[r, c] == Player.WHITE.value:
                plt.text(c + 0.5, r + 0.5, 'W', color='black', ha='center', va='center', fontsize=10, weight='bold')

    output_path = os.path.join(OUTPUT_DIR, "heatmap_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 热力图已生成至 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a hybrid network policy")
    parser.add_argument('--model', required=True, help='Path to a compatible hybrid checkpoint')
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    args = parser.parse_args()
    MODEL_PATH = args.model
    OUTPUT_DIR = args.output_dir
    plot_heatmap()
