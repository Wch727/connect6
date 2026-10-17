# Katagomo 对弈与复盘

安装 NumPy，启动本地页面：

```sh
python -m pip install numpy
python -X utf8 katagomo_ui.py
```

打开 `http://127.0.0.1:8766`。

程序会优先在仓库所在目录、再在其上一级目录中寻找 `Katagomo*` 软件文件夹。也可指定位置：

```sh
python -X utf8 katagomo_ui.py --katagomo-dir "D:/KatagomoConnect6_20250505_OpenCL-only"
```

使用已有的 Windows OpenCL 引擎和 `connectsix19x_b18trans.bin.gz` 权重。引擎、权重由 Katagomo 原软件提供，需保留在本机。本项目通过 GTP 调用它们。

## 操作

- **对弈**：选黑白、设置每颗思考秒数，开始对局。默认 1 秒，可在思考结束后调整。
- **悔棋**：对弈时撤回你最近一颗及后续 AI 回应；复盘时回退一颗。
- **导入棋谱**：选择 SGF 或 JSON。支持 Katagomo 的 19×19 SGF、原网页版 JSON 和逐颗带 `action` 的对局记录。SGF 有变化时读取第一条主变化。
- **逐步复盘**：使用前后按钮、进度条或点击落子记录跳转。点击棋盘会从该处续写，替换原来的后续落子。
- **看看 Katagomo 推荐**：在棋盘上预览当前回合的一颗或两颗推荐。点击“按推荐落子”才改变局面。
- **继续对弈**：复盘到想继续的位置，选好执子，切换“对弈”。后续记录从该处续写。
- **保存**：导出当前整份棋谱为 SGF 或 JSON，包含进度条位置之后尚未播放的落子。

SGF/JSON 导入要求从空棋盘开始，遵循黑方首轮一颗、随后每轮两颗的顺序。HTML 开局分析页是静态局面数据，仍由 `opening_data.py` 处理。

原有 `web_ui.py` 的模型对弈页面使用端口 8765，可以独立运行。关闭这个界面的 Python 服务会结束它启动的引擎进程。
