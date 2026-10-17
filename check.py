import pickle
import os

def inspect_book():
    file_path = "connect6_book.pkl"
    if not os.path.exists(file_path):
        print("❌ 错误：找不到 connect6_book.pkl 文件！")
        return

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        print(f"✅ 成功读取文件！")
        print(f"📊 开局库中共有 {len(data)} 个局面定式。")

        if len(data) == 0:
            print("⚠️ 警告：开局库是空的！请重新运行 build.py。")
            return

        # 打印前 1 个数据样本来看看长什么样
        print("\n--- 数据样本 (Key -> Value) ---")
        first_key = list(data.keys())[0]
        first_val = data[first_key]

        print(f"Key (棋盘状态片段): {first_key[:20]} ... (长度: {len(first_key)})")
        print(f"Value (推荐招法): {first_val}")

        # 统计黑白子
        black_count = first_key.count(1)
        white_count = first_key.count(2)
        print(f"样本分析: 黑子 {black_count} 个, 白子 {white_count} 个")

        # 检查有没有“空棋盘” (全0)
        empty_key = tuple([0] * 361)
        if empty_key in data:
            print("\n✅ 好消息：库里包含【空棋盘】(第一手)的定式！")
        else:
            print("\n⚠️ 坏消息：库里**没有**【空棋盘】数据。")
            print("   -> 这意味着第一手黑棋 AI 会自己思考，如果下偏了，后面就全查不到了。")

    except Exception as e:
        print(f"❌ 读取报错: {e}")

if __name__ == "__main__":
    inspect_book()
