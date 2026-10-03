# 运行环境与本地路径

主 Python 依赖在 `rendering/rgba_pipeline/pyproject.toml` 和 `uv.lock` 中声明。
Blender/Cycles 可使用独立的 Blender 可执行程序，或已有的 Python 3.11 + bpy 4.2.23 runtime。

## 两种 Blender 入口

设置 `RGBA_BLENDER_BIN=/path/to/blender`，或将 Blender 放在 `PATH` 中。
如果没有 Blender 可执行程序，管线会使用以下可配置的 bpy 环境：

| 环境变量 | 默认位置 |
| --- | --- |
| `RGBA_BPY_PYTHON` | `/usr/bin/python3.11` |
| `RGBA_BPY_RUNTIME` | 工作目录的 `runtime/bpy_runtime/` |
| `RGBA_BPY_LIBRARIES` | 工作目录的 `runtime/bpy_x11_runtime/lib/` |

工作目录优先取包含 `runtime/bpy_runtime/` 的上层目录；独立 checkout 时使用 RABench
仓库根目录。镜像读写所需的 `OpenEXR`、`Pillow`、`opencolorio` 必须安装到对应 Python
环境。现有工作区为此保留了管线的 `.runtime/bpy/` 与 `.runtime/host/`，它们是正式本地
依赖，不从临时目录导入。只设置一个存在的 `RGBA_BLENDER_BIN` 时，该可执行程序优先。

## Bash 启动脚本

在交互式 GPU worker 中先执行 `mlx worker login`。三个渲染启动脚本都会激活现有
`anime` Conda 环境；通过已初始化的 Conda、`CONDA_EXE` 或 `RGBA_CONDA_ROOT` 定位安装。
例如可将 `RGBA_CONDA_ROOT` 设为自己的 Miniconda 根目录。参数直接写在各 Python 命令后。

按工作区约定提供以下环境变量；令牌只通过环境传入：

```bash
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_ENDPOINT=http://huggingface-proxy-sg.byted.org
# HF_TOKEN 由本机环境提供
```

## 可选 Luna 标注

渲染本身不需要 Luna。`run_p1.sh` 在渲染后运行三例 Luna 标注，因此这个组合入口还需要
外部 `VLMClient` 与 teacher YAML 配置。默认查找工作目录旁的 AlphaLift checkout，
可用 `RGBA_LUNA_CLIENT` / `RGBA_LUNA_CONFIG`，或 CLI 的 `--client-module` /
`--teacher-config` 指定其他位置。配置中的凭据不提交。

只生成并渲染配对数据时，从管线目录分别执行：

```bash
python -m rgba_pipeline generate --suite p1_pairs_v1 --run-name p1_small_batch --resolution 1024 --samples 512 --seed 42
python -m rgba_pipeline p1-render --run runs/p1_small_batch --max-workers 2 --resume
python -m rgba_pipeline report --run runs/p1_small_batch
```

## 临时文件生命周期

正式源码、配置、提示词、下载资产、运行库和渲染结果均有独立位置。
`codex_tmp/rgba_pipeline/` 只用于下载暂存、进程临时目录和可重建的 Blender 缓存，目录由
调用方按需创建。场景下载器默认在成功解压到 `assets/scenes/` 后删除临时 ZIP；
`--archives` 可显式指定另一个需要保留的下载目录。

任务结束后可清空工作区 `codex_tmp/`。下次运行重新创建临时目录，使用已有正式资产，
不会依赖被清理的安装副本或旧下载分片。需要保留的示例、预览和报告放在工作区的
`outputs/`；现有实验的 `runs/` 保持原位置。
