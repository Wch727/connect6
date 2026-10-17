import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison():
    # 检查文件是否存在
    if not os.path.exists("logs/baseline.csv") or not os.path.exists("logs/ours.csv"):
        print("错误：找不到日志文件！请先运行实验生成 logs/baseline.csv 和 logs/ours.csv")
        return

    # 读取数据
    df_base = pd.read_csv("logs/baseline.csv")
    df_ours = pd.read_csv("logs/ours.csv")

    # 设置画图风格
    plt.style.use('seaborn-v0_8-whitegrid') # 如果报错，改成 plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))

    # 画 Baseline 线 (蓝色，虚线)
    plt.plot(df_base['iteration'], df_base['win_rate'],
             label='Baseline (AlphaZero)',
             color='blue', linestyle='--', linewidth=2, marker='o')

    # 画 Ours 线 (红色，实线)
    plt.plot(df_ours['iteration'], df_ours['win_rate'],
             label='Ours (SL + Rule-Guided)',
             color='red', linewidth=2, marker='s')

    # 装饰图片
    plt.xlabel('Training Iterations', fontsize=12)
    plt.ylabel('Win Rate vs. Heuristic AI', fontsize=12)
    plt.title('Cold-Start Efficiency Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.ylim(-0.05, 1.05) # y轴范围 0-1
    plt.grid(True, alpha=0.3)

    # 保存图片
    os.makedirs("figs", exist_ok=True)
    save_path = "figs/result_chart.png"
    plt.savefig(save_path, dpi=300)
    print(f"图表已生成：{save_path}")
    # plt.show() # 如果在服务器上跑，把这行注释掉

if __name__ == "__main__":
    plot_comparison()
