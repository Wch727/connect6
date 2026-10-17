# Connect6 · 六子棋训练与对弈

基于 AlphaZero 风格的六子棋项目，包含策略价值网络、MCTS、自对弈训练、监督预训练、模型评测与图形界面。

项目源于首届北京市大学生“人工智能+”创新大赛。**“请输入文本”队获得六子棋项目季军（一等奖）。**

## 网页对弈与复盘

先安装 NumPy，再选择一个入口：

```sh
python -m pip install numpy
```

| 入口 | 启动命令 | 地址 | 用途 |
| --- | --- | --- | --- |
| Katagomo | `python -X utf8 katagomo_ui.py` | http://127.0.0.1:8766 | 对弈、推荐落点、SGF/JSON 棋谱导入导出与逐步复盘 |
| 项目 AI | `python -X utf8 web_ui.py` | http://127.0.0.1:8765 | 启发式对弈，或加载本项目训练的模型 |

Katagomo 界面调用本机已有的 Windows OpenCL 引擎与权重，启动时自动查找软件目录，也可通过 `--katagomo-dir` 指定。默认每颗思考 1 秒。见 [Katagomo 使用说明](docs/KATAGOMO.md)。

项目 AI 页面默认使用无需权重的启发式；加载神经网络需要安装 PyTorch，并将权重放入 `checkpoints/`。见 [网页版说明](docs/WEB_UI.md)和[启发式策略](docs/HEURISTIC.md)。

规则：19×19 棋盘，黑方首回合一子，此后每回合两子，横、竖、斜连成六子或以上即胜，无禁手。

## 模型训练与命令行对弈

安装适合本机 CPU 或 CUDA 的 [PyTorch](https://pytorch.org/get-started/locally/)，然后安装训练依赖：

```sh
python -m pip install -r requirements.txt
```

```sh
# 加载已有模型对弈
python -X utf8 play.py --mode human-ai --model checkpoints/model.pt --device cpu

# 小规模自对弈训练
python -X utf8 train.py --model-size small --iterations 2 --self-play-games 2 --num-simulations 20 --checkpoint-dir runs/example

# 从检查点继续训练 20 轮
python -X utf8 train.py --resume checkpoints/checkpoint_iter80.pt --iterations 20 --checkpoint-dir runs/resume80
```

监督预训练入口为 `gen_data.py` 和 `train_sl.py`，实验训练入口为 `train_experiment.py`。详见 [训练说明](docs/TRAINING.md)。KataGomo HTML 局面数据的字段与转换方式见 [数据说明](docs/DATA.md)。当前训练入口支持 CPU/CUDA。

`play.py` 在模型路径不存在时会随机初始化网络；使用已有权重时请确认控制台显示模型已加载。

## 模型评测

```sh
python -X utf8 evaluate.py --model checkpoints/model_a.pt --opponent-model checkpoints/model_b.pt --games 20 --simulations 400 --output logs/match.json
```

双方交换先后手，输出胜负和、得分率与逐局棋谱。默认开启规则辅助，使用 `--no-rule-based` 可对双方关闭。

## 桌面 GUI

```sh
python -m pip install -r requirements-gui.txt
python -X utf8 gui_heuristic.py
python -X utf8 gui_neural.py --model checkpoints/model.pt
```

比赛版、混合版及历史补充赛制界面均保留，用途见 [GUI 说明](docs/GUI.md)。绘图依赖按需安装：`python -m pip install -r requirements-analysis.txt`。

## 目录

| 路径 | 内容 |
| --- | --- |
| `game/` | 棋盘规则、启发式、Katagomo 连接与棋谱读写 |
| `model/`、`mcts/` | 策略价值网络、串行与批量 MCTS |
| `training/` | 自对弈、回放、优化与检查点 |
| `web/` | 两个本地网页界面的静态资源 |
| `docs/` | 界面、训练与数据使用说明 |
| `tests/` | 回归检查 |

模型权重、开局库、原始数据、棋谱、日志和运行结果保存在本地，不随源码分发。开局库可通过 `build.py` 构建，命令见 [数据说明](docs/DATA.md)。

## 测试

```sh
python -X utf8 -B -m unittest discover -s tests -v
```

## 使用许可与引用

仅限非商业学习、教学和科研使用。使用或分发时须注明项目来源，修改后须说明修改内容；学术成果中实质使用本项目时须引用代码仓库及所用版本。商用须事先获得单独书面授权。

完整条款见 [LICENSE.md](LICENSE.md)，代码引用见 [CITATION.md](CITATION.md)。
