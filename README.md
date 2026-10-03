# RABench

Render-Alpha Benchmark：以 Blender/Cycles 构建 RGB/RGBA 数据和可编辑场景。
当前管线支持固定场景配对、PBR 模板以及单物体提取，并保留联合渲染实验入口。

## 目录与工作流

| 位置 | 内容 |
| --- | --- |
| [rendering/rgba_pipeline/](rendering/rgba_pipeline/README.md) | 渲染管线、CLI、版本化预设和资产来源清单 |
| [docs/i2rgba_task_taxonomy.md](docs/i2rgba_task_taxonomy.md) | 当前任务定义、归属与遮挡、边界、半透明及外部干扰分类 |
| [docs/rgba_benchmark_design.md](docs/rgba_benchmark_design.md) | 历史设计参考，保留 GT 和评价方法的背景 |
| [docs/environment.md](docs/environment.md) | 环境、运行路径、可选 Luna 客户端和临时文件约定 |
| [examples/](examples/README.md) | 不依赖下载素材的最小 RGB/RGBA 制作示例 |

主管线的三个常用入口：

- `rendering/rgba_pipeline/scripts/run_p1.sh`：固定 24 对场景 RGB / 中性光 RGBA，随后运行三例 Luna 标注。
- `rendering/rgba_pipeline/scripts/run_template_pbr.sh`：PBR 对象与原生场景模板的联合渲染。
- `rendering/rgba_pipeline/scripts/run_isolated_objects.sh`：提取现有场景配置中的单个语义物体。

## 快速开始

从本 Git 仓库根目录执行：

```bash
cd rendering/rgba_pipeline
uv sync --frozen --extra dev
export RGBA_BLENDER_BIN=/path/to/blender
uv run python -m rgba_pipeline --help
```

准备素材、生成场景和渲染的完整命令见[管线说明](rendering/rgba_pipeline/README.md)。
在现有工作目录中，先进入交互式 GPU worker 并激活 `anime`，也可直接使用上述 Bash 入口。
GPU 渲染、素材下载和 Luna 调用按需执行；查看 CLI 不会启动它们。

## 本地文件

源码、预设、提示词和小型资产来源清单随仓库保存。下载资产、渲染结果、运行库、
环境探测结果和凭据由忽略规则排除。现有资产和结果仍分别位于管线的 `assets/` 与 `runs/`。

工作区的 `codex_tmp/` 只用于可重建的下载暂存与进程临时文件。清空其中已结束任务的文件，
不会改变正式素材或运行环境。保留的示例图片、预览与报告位于工作区的 `outputs/`。
