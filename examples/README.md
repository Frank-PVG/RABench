# 最小 RGB/RGBA 示例

`rgba_pair_example.py` 使用程序生成的薄纱与背景演示透明底渲染、线性合成和配对导出。
它使用 CPU，不下载模型或素材；不模拟真实玻璃折射，也不代替主渲染管线。

需要 Blender 4.2，以及普通 Python 环境中的 `numpy`、`opencv-python-headless`。
从本 Git 仓库根目录运行：

```bash
python examples/rgba_pair_example.py --blender /path/to/blender --output-dir outputs/rgba_pair --resolution 512 --samples 64 --opacity 0.35 --seed 42
```

脚本将输出 RGB、RGBA、alpha、背景、预览与元数据。已有工作目录中的示例结果放在
外层 `outputs/examples/rgba_pair/`。`--render-only` 与 `--package-only` 支持分开执行
渲染和配对。
