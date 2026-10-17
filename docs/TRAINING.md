# 训练与评测

## 监督预训练

```sh
python -X utf8 gen_data.py
python -X utf8 train_sl.py
python -X utf8 train_experiment.py --model-size hybrid --pretrained checkpoints/sl_model.pt --log-name bc_rules.csv --iterations 50 --eval-games 10 --eval-simulations 400 --eval-no-rule-based
```

`gen_data.py` 使用启发式策略生成 3000 局，每 500 局保存一批，并进行八种对称增强。生成规模在脚本入口调整。

`train_sl.py` 加载 `data/` 下的 `.pkl` 文件，数据项格式为 `(state, policy, value)`。默认使用 Hybrid 网络、20 epochs、batch 512，输出 `checkpoints/sl_model.pt`。脚本记录训练总损失；CSV 中的分项损失列为固定占位值。数据集由调用者划分，脚本直接训练所加载的全部数据。

`build.py` 和 `gen_HTML_data.py` 读取 KataGomo 六子棋 AI 的 HTML 局面分析数据，生成开局库或监督样本。候选组内均匀生成策略标签，价值标签取待落子玩家视角的引擎评估。来源、格式与转换命令见 [数据说明](DATA.md)。

## 实验训练

`train_experiment.py --pretrained PATH` 载入网络参数，`--model-size` 须与权重结构匹配。默认每轮 20 局、每步 400 次模拟、5 epochs。实验权重保存到 `checkpoints_exp/`，包含模型参数和迭代号。不同实验使用不同 `--log-name`，同名 CSV 会被覆盖。

`run.sh` 在 Bash 中依次运行 random、rules-only、BC-only、BC+rules 四组训练，评测统一关闭规则辅助。`plot_results.py` 绘制 baseline 与 ours 两组曲线。Windows 下可逐条运行脚本内的 Python 命令。

## 断点续训

```sh
python -X utf8 train.py --resume checkpoints/checkpoint_iter80.pt --iterations 20 --checkpoint-dir runs/resume80
```

`--iterations M` 表示额外执行 M 轮，从检查点完成的 K 轮继续到 K+M。主训练器检查点用于恢复模型、优化器、调度器和训练进度；监督预训练文件用于初始化网络。使用 `--checkpoint-dir` 为运行指定独立目录。

## 评测

```sh
python -X utf8 evaluate.py --model checkpoints/model_a.pt --opponent-model checkpoints/model_b.pt --games 20 --simulations 400 --output logs/match.json
```

支持 random、greedy、self 对手，以及 `--opponent-model` 指定的模型。模型对手使用 MCTS，关闭探索噪声并取最大概率动作。`--no-rule-based` 对双方关闭规则辅助。各组比较使用相同的评测设置。

评测从空棋盘开始，逐局交换执黑执白，输出胜负和、得分率及落子记录。固定设置下的确定性对局会重复。共享评测函数在结束时恢复 Python/NumPy 随机状态和网络训练模式。

## 实现约定

- 单次动作放置一颗棋子，动作编号为 `row * 19 + col`；黑方首回合一子，此后每回合两子。价值回传按玩家身份处理符号。
- 输入为 17 通道：双方各八个棋盘平面和一个常数平面。历史列表为当前棋盘加最近七个存储棋盘；使用旧权重时保持对应输入约定。
- 战术扫描作用于搜索根部；检测到获胜续着时直接出招，否则限制防守候选。普通搜索执行配置的模拟次数。
- 设备入口为 CPU/CUDA。NPU 训练适配需单独实现。
