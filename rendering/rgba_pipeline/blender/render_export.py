"""Scene-linear EXR and straight-alpha display exports."""
import json
from pathlib import Path
import bpy
import numpy as np
import OpenEXR
from PIL import Image

def display_rgb(rgb: np.ndarray, cfg: dict) -> np.ndarray:
    """View transform is for display only; no transform touches the EXR targets."""
    values = np.ascontiguousarray(rgb, dtype=np.float32) * (2.0 ** float(cfg.get('exposure', 0.0)))
    view = cfg.get('view_transform', 'Standard')
    if view in {'Standard', 'Raw'}:
        if view == 'Standard':
            values = np.where(values <= .0031308, 12.92 * values, 1.055 * np.maximum(values, 0) ** (1 / 2.4) - .055)
    else:
        import PyOpenColorIO as ocio
        config_path = Path(bpy.utils.resource_path('LOCAL')) / 'datafiles/colormanagement/config.ocio'
        config = ocio.Config.CreateFromFile(str(config_path))
        source = config.getColorSpace('scene_linear').getName()
        processor = config.getProcessor(source, 'sRGB', view, ocio.TRANSFORM_DIR_FORWARD).getDefaultCPUProcessor()
        processor.applyRGB(values.reshape(-1, 3))
    return np.round(np.clip(values, 0, 1) * 255).astype(np.uint8)


def save_render_buffers(out: Path, cfg: dict) -> None:
    """Write the renderer's associated-alpha EXR directly, and derive previews.

OpenEXR avoids the ambiguous Blender image reload/unpremultiply round trip.
PNG is straight RGBA; alpha previews remain linear, not display transformed.
"""
    out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.exr_codec = 'ZIP'
    destination = out / 'linear_premult.exr'
    bpy.data.images['Render Result'].save_render(str(destination), scene=scene)
    with OpenEXR.File(str(destination)) as source:
        rgba = np.array(source.channels()['RGBA'].pixels, dtype=np.float32, copy=True)
    alpha = np.clip(rgba[..., 3], 0, 1)
    # Match the existing transparent-pixel convention used by basic_checks.
    transparent = alpha < 1e-4
    rgba[transparent] = 0
    alpha[transparent] = 0
    OpenEXR.File({}, {'RGBA': rgba}).write(str(destination))
    OpenEXR.File({}, {'RGB': np.repeat(alpha[..., None], 3, axis=2)}).write(str(out / 'alpha.exr'))
    straight = np.divide(rgba[..., :3], alpha[..., None], out=np.zeros_like(rgba[..., :3]), where=alpha[..., None] > 1e-8)
    png = np.concatenate([display_rgb(straight, cfg), np.round(alpha[..., None] * 255).astype(np.uint8)], axis=2)
    Image.fromarray(png).save(out / 'straight_rgba.png')
    Image.fromarray(np.round(alpha * 255).astype(np.uint8)).save(out / 'alpha_preview.png')
    h, w = alpha.shape
    yy, xx = np.mgrid[:h, :w]
    checker = np.where(((xx // 32 + yy // 32) % 2)[..., None], .08, .6).astype(np.float32)
    for background, name in [(0.0, 'preview_black'), (1.0, 'preview_white'), (checker, 'preview_checker')]:
        composite = rgba[..., :3] + (1 - alpha[..., None]) * background
        Image.fromarray(display_rgb(composite, cfg)).save(out / (name + '.png'))
    if cfg.get('context_output'):
        Image.fromarray(display_rgb(rgba[..., :3], cfg)).save(out / 'rgb.png')
    (out / 'image_conventions.json').write_text(json.dumps({
        'exr': 'scene-linear float32 premultiplied RGB and linear alpha',
        'png': 'straight RGBA; display RGB and linear alpha',
        'view_transform': cfg.get('view_transform', 'Standard'), 'exposure': cfg.get('exposure', 0.0),
    }, indent=2) + '\n')
