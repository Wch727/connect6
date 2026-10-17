# GUI 实现说明

六个桌面界面分别用于启发式、神经网络和比赛模式。

| 入口 | 实现与用途 | 模型/数据 |
| --- | --- | --- |
| `gui_heuristic.py` | 独立启发式引擎、人机对弈与比赛操作界面 | 不需要神经网络权重 |
| `gui2.py` | 内置启发式/威胁搜索，多种对局模式；可选 Numba 加速 | 不需要神经网络权重 |
| `gui_neural.py` | 神经网络 MCTS 对弈，支持 `--model` 指定权重 | PyTorch、兼容检查点 |
| `gui_competition.py` | 比赛客户端：模型选择、开局库、计时、悔棋及棋谱/复盘相关操作 | PyTorch；开局库为本地可选资产 |
| `gui_hybrid.py` | 另一套比赛/混合操作界面，包含开局库和模型控制 | PyTorch；可选本地开局库 |
| `gui_additional.py` | 历史人机交替协作模式 | 对应后来取消的补充赛制，作为历史实现保留 |

公共部分采用 Pygame 绘制棋盘与控件，通过事件循环处理交互，模型客户端使用后台线程执行搜索。神经网络客户端通过 `Connect6Player` 调用网络与 MCTS；独立启发式界面使用自己内置的评分/威胁逻辑。

各界面分别维护计时和棋谱逻辑。`gui_competition.py` 的 `TIME_LIMIT` 为 300000 秒，`gui_heuristic.py` 为 600 秒；比赛时按赛事要求设置计时和搜索预算。`gui_neural.py` 支持自动选择模型，复现实验使用 `--model` 指定权重。

开局库从本地 `connect6_book.pkl` 加载，源码仓库提供构建工具。每下一子后重新查询新局面，表中同组坐标作为可选落点。模型对比时统一开局库设置。中文字体由本机字体提供，桌面窗口使用固定尺寸。

## 网页版

`python -X utf8 web_ui.py` 启动本地浏览器界面，独立于以上六个历史 GUI，不需要 Pygame。使用说明见 [网页版说明](WEB_UI.md)。

`python -X utf8 katagomo_ui.py` 启动 Katagomo 对弈与复盘页面，使用本机 OpenCL 引擎，支持推荐落点和 SGF/JSON 导入导出。见 [Katagomo 说明](KATAGOMO.md)。
