#!/usr/bin/env python3
"""独立示例：渲染一块半透明薄纱，再构造 RGB 输入与 RGBA 参考图。

用途
----
演示本项目已有的“透明底渲染 → 线性颜色合成 → 配对导出”思路。
不依赖项目中的数据、模型或其他 Python 文件；背景由程序生成。
这是简化的薄纱素材示例，不模拟真实玻璃折射，也不代替独立渲染质检。

依赖与运行
----------
需要 Blender 4.2，以及普通 Python 环境中的 numpy、opencv-python-headless。
Blender 必须能从命令行启动，或通过 --blender 指定其可执行程序。

    python rgba_pair_example.py --output-dir ./example_output \
        --resolution 512 --samples 64 --opacity 0.35 --seed 42

    python rgba_pair_example.py --blender /path/to/blender \
        --output-dir ./example_output --resolution 512 --samples 64

示例默认使用 CPU。参数直接放在命令后，方便修改；不会下载模型或素材。
只有前景参与渲染，背景在渲染后合成，因此背景不会进入 alpha。

输出
----
target_rgba.png       16-bit RGBA 前景，RGB 为 sRGB 编码、straight alpha。
alpha.png            单独的 16-bit alpha，数值不做 sRGB 编码。
input_rgb.png        加入背景后的 RGB 输入。
background.png       本次使用的背景。
alpha_preview.png    便于直接查看的 8-bit alpha。
checker_preview.png  棋盘底合成预览；棋盘不会写入目标 RGBA。
metadata.json        描述、颜色约定、参数和相对文件名。

渲染和打包使用两个进程：Blender 自带 Python 无需安装 OpenCV。
--render-only / --package-only 也允许在不同 Python 环境中分别执行两步。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--opacity", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render-only", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--package-only", action="store_true", help="仅打包已经渲染的前景")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    if args.resolution < 32 or args.samples < 1 or args.threads < 1:
        parser.error("resolution 至少为 32，samples 和 threads 至少为 1")
    if not 0.0 < args.opacity < 1.0:
        parser.error("此半透明示例要求 0 < opacity < 1")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def render_target(args: argparse.Namespace) -> None:
    """由 Blender 执行；不依赖普通 Python 环境中的 OpenCV。"""
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.seed = args.seed
    scene.cycles.transparent_max_bounces = 16
    scene.cycles.use_denoising = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = args.threads
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "16"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    # 灯光仍影响布料，但照明环境不作为可见背景输出。
    world = bpy.data.worlds.new("Lighting environment")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.8, 0.85, 1, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.25
    scene.world = world

    # 用弯曲网格代替外部布料资产，便于独立运行和理解。
    count = 49
    vertices = []
    for row in range(count):
        v = 2.0 * row / (count - 1) - 1.0
        for column in range(count):
            u = 2.0 * column / (count - 1) - 1.0
            vertices.append((
                1.6 * u,
                0.22 * math.sin(3.0 * math.pi * u) + 0.08 * math.sin(4.0 * v),
                0.85 * v - 0.18 * (1.0 - u * u),
            ))
    faces = []
    for row in range(count - 1):
        for column in range(count - 1):
            a = row * count + column
            faces.append((a, a + 1, a + count + 1, a + count))
    mesh = bpy.data.meshes.new("Wavy fabric mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    fabric = bpy.data.objects.new("Sheer blue fabric", mesh)
    scene.collection.objects.link(fabric)
    for polygon in mesh.polygons:
        polygon.use_smooth = True

    # 白色 Transparent BSDF 不改变背景颜色/位置，适合演示普通 alpha。
    # 本例的“透光”来自透明分量，不以折射玻璃材质冒充普通 RGBA。
    material = bpy.data.materials.new("Simple sheer material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    surface = nodes.new("ShaderNodeBsdfDiffuse")
    surface.inputs["Color"].default_value = (0.14, 0.38, 0.68, 1.0)
    surface.inputs["Roughness"].default_value = 0.4
    mix = nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = args.opacity
    links = material.node_tree.links
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(surface.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    mesh.materials.append(material)

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 4.2
    camera = bpy.data.objects.new("Camera", camera_data)
    camera.location = (0.5, -5.0, 0.3)
    camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(camera)
    scene.camera = camera
    light_data = bpy.data.lights.new("Soft key light", "AREA")
    light_data.energy = 220
    light_data.size = 4
    light = bpy.data.objects.new("Soft key light", light_data)
    light.location = (-2.0, -3.0, 4.0)
    light.rotation_euler = (-Vector(light.location)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(light)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output_dir / "target_rgba.png")
    bpy.ops.render.render(write_still=True)
    settings = {
        "renderer": "Blender Cycles",
        "blender_version": bpy.app.version_string,
        "resolution": args.resolution,
        "samples": args.samples,
        "opacity": args.opacity,
        "seed": args.seed,
        "view_transform": "Standard",
        "device": "CPU",
    }
    (args.output_dir / "render_metadata.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def package_sample(output_dir: Path) -> None:
    """由普通 Python 执行；PNG 中的 alpha 不经过颜色空间转换。"""
    import cv2
    import numpy as np

    target = cv2.imread(str(output_dir / "target_rgba.png"), cv2.IMREAD_UNCHANGED)
    if target is None or target.ndim != 3 or target.shape[2] != 4:
        raise ValueError("需要先渲染四通道 target_rgba.png")
    if target.dtype != np.uint16:
        raise ValueError("示例约定目标 RGBA 为 16-bit PNG")
    rgba = cv2.cvtColor(target, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 65535.0
    alpha = rgba[..., 3:4]
    if float(alpha.max()) <= 0.001:
        raise ValueError("前景为空，需检查场景或相机")

    def srgb_to_linear(value):
        return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(value):
        value = np.clip(value, 0, 1)
        return np.where(value <= 0.0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - 0.055)

    def save(name, values, color=False, bits=8):
        dtype, maximum = (np.uint16, 65535) if bits == 16 else (np.uint8, 255)
        encoded = np.round(np.clip(values, 0, 1) * maximum).astype(dtype)
        if color:
            encoded = cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output_dir / name), encoded):
            raise OSError(f"无法保存 {name}")

    height, width = alpha.shape[:2]
    y, x = np.indices((height, width), dtype=np.float32)
    tile = max(4, width // 12)
    checker = ((x // tile + y // tile) % 2)[..., None]
    background = np.stack((
        0.72 + 0.16 * x / max(width - 1, 1),
        0.55 + 0.22 * y / max(height - 1, 1),
        0.34 + 0.16 * x / max(width - 1, 1),
    ), axis=-1)
    # 将实际保存的 8-bit 背景作为合成输入，确保文件之间可以直接复核。
    background = np.round(background * 255.0).astype(np.uint8).astype(np.float32) / 255.0
    foreground_linear = srgb_to_linear(rgba[..., :3])

    def composite(background_srgb):
        # straight RGB 只乘一次 alpha；透明度本身保持线性数值。
        return linear_to_srgb(
            foreground_linear * alpha + srgb_to_linear(background_srgb) * (1.0 - alpha)
        )

    save("background.png", background, color=True)
    save("input_rgb.png", composite(background), color=True)
    save("alpha.png", alpha[..., 0], bits=16)
    save("alpha_preview.png", alpha[..., 0])
    save("checker_preview.png", composite(np.where(checker, 0.75, 0.3)), color=True)
    metadata = {
        "description": "一块浅蓝色半透明薄纱，有波浪褶皱，保留布料整体与透明度。",
        "input_rgb": "input_rgb.png",
        "target_rgba": "target_rgba.png",
        "alpha": "alpha.png",
        "background": "background.png",
        "rgb_encoding": "sRGB PNG; straight alpha",
        "compositing": "C = F * alpha + B * (1-alpha), in linear RGB",
        "alpha_semantics": "仅表示布料的不透明度，不包含背景或外部地面阴影",
        "soft_alpha_fraction": float(np.mean((alpha > 0.01) & (alpha < 0.99))),
        "render": json.loads((output_dir / "render_metadata.json").read_text(encoding="utf-8")),
        "scope": "最小数据制作示例；未进行真实折射或 Core 独立底图渲染验收",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "SUCCESS", "output_dir": str(output_dir),
                      "soft_alpha_fraction": metadata["soft_alpha_fraction"]}, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.render_only:
        render_target(args)
        return
    if not args.package_only:
        subprocess.run([
            args.blender, "--background", "--factory-startup", "--python-exit-code", "1",
            "--python", str(Path(__file__).resolve()), "--", "--render-only",
            "--output-dir", str(args.output_dir), "--resolution", str(args.resolution),
            "--samples", str(args.samples), "--opacity", str(args.opacity),
            "--seed", str(args.seed), "--threads", str(args.threads),
        ], check=True)
    package_sample(args.output_dir)


if __name__ == "__main__":
    main()
