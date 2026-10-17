# 数据来源与格式

## 示范来源

数据工具支持两类示范：

- `gen_data.py`：本项目 `HeuristicAI` 生成的自对弈决策与最终胜负。
- HTML 开局数据：[KataGomo / KataGoMo](https://github.com/hzyhhzy/KataGomo) 六子棋 AI 的局面分析库，用于构建开局查询表和策略示范。

HTML 数据来自 `Connect6_19x19_maxmoves109_openings` 与 `Connect6_19x19.MoveLimit113.Openings` 两套开局库。每个局面页面保存棋盘及候选分析，入口跳转页用于导航。HTML 文件数、完整对局数和训练样本数分别统计。

KataGomo 是第三方 AI。其程序、权重和开局数据的授权由各自权利人提供；本仓库的许可证仅覆盖本仓库有权授权的内容。原始 HTML 与打包程序由用户单独准备。

## HTML 字段

| 字段 | 含义 |
| --- | --- |
| `board` | 按行展开的棋盘，0 为空、1 为黑、2 为白 |
| `nextPla` | 下一颗棋子的落子方，1 为黑、2 为白 |
| `moves` | 当前局面的候选分析列表 |
| `xy` | 同一候选动作的对称等价落点，坐标格式为 `[x, y]` |
| `p` | 候选动作的策略先验 |
| `v` | 候选动作的访问次数 |
| `wl` | 白方视角的胜负评估值，范围 [-1, 1] |
| `links` / `linkSyms` | 落子后的局面链接及对称变换信息 |

`xy` 中每个坐标都是独立可选的一颗棋子，不是一个回合内的连续落子序列。消费开局库时选择一个落点，执行后根据新棋盘继续查询。

原导出的 `book.js` 以 `y * board_size + x` 索引棋盘；对应 NumPy 坐标为 `row=y, col=x`。源码 `cpp/book/book.cpp` 在生成 `xy` 时遍历局面的等价对称变换。

## 转换为训练样本

`build.py` 和 `gen_HTML_data.py` 共用 `opening_data.py`。解析器支持原导出中的尾逗号，过滤越界、占用和重复坐标，再从有访问记录的候选组中选择访问次数最多的一组。

策略标签在该组的等价落点间均匀分配概率。价值标签转换为待落子玩家视角：黑方待落时为 `-wl`，白方待落时为 `wl`。这是教师引擎的评估标签。

静态页面提供当前棋盘，历史特征沿用开局数据的重复局面约定。`build.py` 每个局面输出一条训练样本，并保存可选坐标列表；`gen_HTML_data.py` 对同一样本执行八种对称增强。相同棋盘按输入目录及排序后的文件顺序保留首条，避免重复导出。

两个入口默认保留全部合法局面。`--min-value` 按待落子玩家视角设置最低教师评估，范围为 [-1, 1]。

## 使用

```sh
python -X utf8 build.py --html-dirs raw_openings --output-book runs/opening_check/book.pkl --output-data runs/opening_check/samples.pkl --limit 1000
python -X utf8 gen_HTML_data.py --html-dir raw_openings --output runs/opening_check/augmented.pkl --limit 1000
```

`--limit` 限制扫描的 HTML 文件数。输出文件已存在时程序停止，覆盖使用 `--overwrite`。两个入口将选中样本收集到内存后写出，八倍增强会相应增加内存和输出体积。

旧转换文件中的固定 `0.5` 标签不会自动更新。重新训练时使用新版转换器输出的数据。

## GUI 查库

比赛、混合与补充赛制界面将表中坐标作为可选落点处理，过滤已占用位置后选择一子。落子后根据新的棋盘重新查库，未命中时使用模型搜索。
