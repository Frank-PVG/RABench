"""Compatibility imports for existing Blender entrypoints."""
from render_paths import resolve_path
from scene_templates import load_template
from pbr_materials import apply_material_preset, multiply_socket, set_scalar
from geometry_support import apply_transform, slot_bounds, check_placement
from render_export import display_rgb, save_render_buffers
