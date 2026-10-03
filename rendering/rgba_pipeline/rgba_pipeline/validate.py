from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import OpenEXR
from PIL import Image, ImageFilter

from .io import read_json, write_json


def read_exr(path: Path) -> np.ndarray:
    with OpenEXR.File(str(path)) as infile:
        channels = infile.channels()
        if "RGBA" in channels: return np.asarray(channels["RGBA"].pixels, dtype=np.float32)
        rgb = np.asarray(channels["RGB"].pixels, dtype=np.float32)
        return rgb


def _image_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a diagnostic-only mask; it is never part of the RGBA datum."""
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def _over(back: np.ndarray, front: np.ndarray) -> np.ndarray:
    """Alpha-over two linear premultiplied RGBA images."""
    result = np.empty_like(back)
    result[..., :3] = front[..., :3] + (1.0 - front[..., 3:4]) * back[..., :3]
    result[..., 3] = front[..., 3] + (1.0 - front[..., 3]) * back[..., 3]
    return result


def interaction_checks(sample_root: Path, recipe: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    """Compare Joint with the explicitly ordered independent rendering.

    The result is an interaction diagnostic, never an alpha target.  A visible
    residual is expected for contact shadows, indirect colour, and coverage
    changes, while a missing independent layer is a hard data-integrity error.
    """
    errors: list[str] = []
    metrics: dict[str, float] = {}
    order = recipe.get("diagnostics", {}).get("independent_order", [])
    if not order:
        return errors, metrics
    composed: np.ndarray | None = None
    for object_id in order:
        path = sample_root / "variants" / "independent" / "independent" / object_id / "linear_premult.exr"
        if not path.exists():
            return [f"missing independent layer for {object_id}"], metrics
        layer = read_exr(path)
        composed = layer if composed is None else _over(composed, layer)
    joint = read_exr(sample_root / "joint" / "linear_premult.exr")
    diff = np.abs(joint[..., :3] - composed[..., :3]).mean(axis=2)
    visible = (joint[..., 3] > 0.01) | (composed[..., 3] > 0.01)
    roi = visible & (diff > 0.003)
    metrics["joint_independent_mae_visible"] = float(diff[visible].mean()) if visible.any() else 0.0
    metrics["joint_independent_effect_fraction"] = float(roi.mean())
    diagnostics = sample_root / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    _write_mask(diagnostics / "effect_roi.png", roi)
    _write_mask(diagnostics / "visibility_valid.png", visible)
    return errors, metrics


def basic_checks(sample_root: Path) -> tuple[list[str], dict[str, float]]:
    errors: list[str] = []; metrics: dict[str, float] = {}
    joint = sample_root / "joint"
    required = [joint / "linear_premult.exr", joint / "alpha.exr", joint / "straight_rgba.png", joint / "preview_black.png", joint / "preview_white.png"]
    for file in required:
        if not file.exists(): errors.append(f"missing {file.relative_to(sample_root)}")
    if errors: return errors, metrics
    p = read_exr(joint / "linear_premult.exr"); alpha = read_exr(joint / "alpha.exr")[:, :, 0]
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(alpha)): errors.append("non-finite EXR value")
    metrics["alpha_min"] = float(alpha.min()); metrics["alpha_max"] = float(alpha.max()); metrics["alpha_nonzero_fraction"] = float(np.mean(alpha > .001))
    if alpha.min() < -.001 or alpha.max() > 1.001: errors.append("alpha outside [0,1]")
    if metrics["alpha_nonzero_fraction"] < .01: errors.append("target is nearly fully transparent")
    if metrics["alpha_nonzero_fraction"] > .98: errors.append("target unexpectedly fills frame")
    transparent = alpha < 1e-4
    if transparent.any():
        metrics["transparent_p_abs_max"] = float(np.abs(p[:, :, :3][transparent]).max())
        if metrics["transparent_p_abs_max"] > .02: errors.append("nonzero premult color in fully transparent region")
    png = _image_array(joint / "straight_rgba.png")
    if png.shape[:2] != alpha.shape: errors.append("PNG/EXR dimensions disagree")
    return errors, metrics


def core_checks(sample_root: Path) -> tuple[list[str], dict[str, float]]:
    errors, metrics = basic_checks(sample_root)
    if errors: return errors, metrics
    p = read_exr(sample_root / "joint" / "linear_premult.exr")[:, :, :3]
    alpha = read_exr(sample_root / "joint" / "alpha.exr")[:, :, 0]
    residuals=[]; p95=[]; outliers=[]
    for probe in ("black", "white", "red", "blue", "checker_fine", "checker_coarse", "noise", "stripes"):
        actual_file = sample_root / "probes" / probe / "core_probe" / "linear_premult.exr"
        bg_file = sample_root / "probes" / probe / "core_backplate" / "linear_premult.exr"
        if not actual_file.exists() or not bg_file.exists():
            errors.append(f"missing Core probe render for {probe}"); continue
        actual = read_exr(actual_file)[:, :, :3]; background = read_exr(bg_file)[:, :, :3]
        fit = np.abs(actual - (p + (1 - alpha[:, :, None]) * background)).mean(axis=2)
        residuals.append(float(fit.mean())); p95.append(float(np.percentile(fit,95))); outliers.append(float(np.mean(fit > .02)))
    if residuals:
        metrics.update({"core_fit_mae": float(np.mean(residuals)), "core_fit_p95": float(np.max(p95)), "core_outlier_fraction": float(np.max(outliers))})
        # First-pass fixed gates. The optional noise estimator is intentionally reported separately.
        if metrics["core_fit_mae"] > .002: errors.append("Core mean compositing residual > 0.002")
        if metrics["core_fit_p95"] > .01: errors.append("Core P95 compositing residual > 0.01")
        if metrics["core_outlier_fraction"] > .01: errors.append("Core outlier fraction > 1%")
    return errors, metrics


def physical_checks(sample_root: Path) -> tuple[list[str], dict[str, float]]:
    errors, metrics = basic_checks(sample_root)
    effects: list[float] = []
    checker_effect: np.ndarray | None = None
    for probe in ("black", "white", "red", "blue", "checker_fine", "checker_coarse", "noise", "stripes"):
        c = sample_root / "physical" / probe / "physical_full" / "linear_premult.exr"
        b = sample_root / "physical" / probe / "without_target" / "linear_premult.exr"
        if not c.exists() or not b.exists():
            errors.append(f"missing Physical C/B pair for {probe}")
            continue
        delta = np.abs(read_exr(c)[..., :3] - read_exr(b)[..., :3]).mean(axis=2)
        # This is evidence that the C/B pair came from actual re-renders.  It
        # intentionally does not claim that C-B is an alpha mask or oracle.
        effects.append(float(np.mean(delta > .01)))
        if probe == "checker_fine":
            checker_effect = delta > .01
    if effects:
        metrics["physical_cb_effect_fraction_max"] = max(effects)
        if metrics["physical_cb_effect_fraction_max"] < .005:
            errors.append("Physical C/B pairs have no detectable target/background effect")
    alpha = read_exr(sample_root / "joint" / "alpha.exr")[:, :, 0]
    diagnostics = sample_root / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    _write_mask(diagnostics / "visibility_valid.png", alpha > .001)
    if checker_effect is not None:
        _write_mask(diagnostics / "effect_roi.png", checker_effect)
    return errors, metrics


def validate_sample(sample_root: Path) -> dict[str, Any]:
    recipe = read_json(sample_root / "scene_recipe.json")
    if recipe.get('isolation'):
        from .isolation import validate_isolated_output
        return validate_isolated_output(sample_root, recipe)
    is_core = recipe["background_protocol"]["mode"] == "core"
    errors, metrics = core_checks(sample_root) if is_core else physical_checks(sample_root)
    interaction_errors, interaction_metrics = interaction_checks(sample_root, recipe) if is_core else ([], {})
    errors.extend(interaction_errors); metrics.update(interaction_metrics)
    if recipe["identity"]["template_id"] == "neutral_veil":
        alpha = read_exr(sample_root / "joint" / "alpha.exr")[:, :, 0]
        fractional = (alpha >= .05) & (alpha <= .95)
        metrics["fractional_alpha_fraction"] = float(fractional.mean())
        # The woven sheets create physical sub-pixel coverage.  This check is
        # deliberately global; geometry_debug remains the source for semantic
        # support masks, rather than pretending it is continuous alpha.
        if metrics["fractional_alpha_fraction"] < .01:
            errors.append("veil sample lacks at least 1% continuous-alpha coverage")
    status = "pass" if not errors else "fail"
    representation = "core_verified" if is_core and status == "pass" else "core_candidate" if is_core else "physical"
    report = {"sample_id": recipe["identity"]["sample_id"], "qa_status": status, "representation_class": representation, "errors": errors, "metrics": metrics, "renderer_alpha_semantics": "core_effective_alpha" if is_core else "renderer_alpha_only"}
    write_json(sample_root / "qa.json", report); return report


def calibration_checks(run: Path) -> dict[str, Any]:
    experiment = read_json(run / "experiment.json") if (run / "experiment.json").exists() else {}
    if experiment.get("suite") == "template_pbr_v1":
        # Reuse the established empty-film and opaque-target gates. The small
        # template suite contains no veil/coverage-film calibration specimens.
        checks: list[dict[str, Any]] = []
        errors: list[str] = []
        empty = run / "calibration" / "empty_core" / "alpha.exr"
        if empty.exists():
            peak = float(np.max(np.abs(read_exr(empty)[:, :, 0])))
            checks.append({"name": "empty_core_background", "max_abs_alpha": peak, "pass": peak <= 1e-5})
        else:
            errors.append("missing empty Core calibration render")
        for path in sorted((run / "recipes").glob("*.json")):
            recipe = read_json(path)
            if recipe["background_protocol"]["mode"] != "core":
                continue
            sid = recipe["identity"]["sample_id"]
            opaque = run / "samples" / sid / "joint" / "alpha.exr"
            if not opaque.exists():
                errors.append(f"missing opaque calibration render {sid}")
                continue
            a = read_exr(opaque)[:, :, 0]; support = a > .99
            err = float(np.max(np.abs(a[support] - 1.0))) if support.any() else float("inf")
            checks.append({"name": f"opaque_target_alpha_{sid}", "support_fraction": float(support.mean()),
                           "max_abs_error": err, "pass": bool(err <= .02)})
        errors.extend(check["name"] for check in checks if not check["pass"])
        result = {"schema_version": "calibration_template_v1", "status": "pass" if not errors else "fail",
                  "checks": checks, "errors": errors,
                  "not_applicable": ["veil_alpha_over", "thin_film_effective_opacity"]}
        write_json(run / "calibration.json", result)
        return result
    expansion_anchor = run / "recipes" / "SC01_L1_C1.json"
    if expansion_anchor.exists():
        checks: list[dict[str, Any]] = []
        errors: list[str] = []
        empty = run / "calibration" / "empty_core" / "alpha.exr"
        if empty.exists():
            a = read_exr(empty)[:, :, 0]
            peak = float(np.max(np.abs(a)))
            checks.append({"name": "empty_core_background", "max_abs_alpha": peak, "pass": peak <= 1e-5})
        else:
            errors.append("missing empty Core calibration render")
        opaque_root = run / "samples" / "SC01_L1_C1"
        opaque_path = opaque_root / "joint" / "alpha.exr"
        if opaque_path.exists():
            a = read_exr(opaque_path)[:, :, 0]; support = a > .99
            err = float(np.max(np.abs(a[support] - 1.0))) if support.any() else float("inf")
            checks.append({"name": "opaque_target_alpha", "support_fraction": float(support.mean()),
                           "max_abs_error": err, "pass": bool(err <= .02)})
        else:
            errors.append("missing SC01_L1_C1 opaque calibration render")
        veil_ids = sorted(p.stem for p in (run / "recipes").glob("TR*_L1_C1.json"))
        veil_sample = next((sid for sid in ["TR01_L1_C1", "TR04_L1_C1", *veil_ids]
                            if (run / "samples" / sid / "scene_recipe.json").exists()), None)
        if veil_sample:
            root = run / "samples" / veil_sample
            recipe = read_json(root / "scene_recipe.json")
            order = recipe["diagnostics"]["independent_order"]
            expected: np.ndarray | None = None
            for object_id in order:
                path = root / "variants" / "independent" / "independent" / object_id / "alpha.exr"
                if not path.exists(): errors.append(f"missing calibration layer {veil_sample}/{object_id}"); expected = None; break
                layer = read_exr(path)[:, :, 0]
                expected = layer if expected is None else layer + (1.0 - layer) * expected
            joint = root / "joint" / "alpha.exr"
            if expected is not None and joint.exists():
                actual = read_exr(joint)[:, :, 0]; residual = np.abs(actual - expected)
                checks.append({"name": f"alpha_over_{veil_sample}", "mae": float(residual.mean()),
                               "p95": float(np.percentile(residual, 95)),
                               "pass": bool(residual.mean() <= .02 and np.percentile(residual, 95) <= .1)})
            elif expected is not None: errors.append(f"missing joint calibration render {veil_sample}")
        else:
            errors.append("no rendered translucent expansion sample is available for alpha-over calibration")
        film_sample = next((sid for sid in ("TR11_L1_C1", "TR12_L1_C1", "TR13_L1_C1", "TR14_L1_C1", "TR15_L1_C1")
                            if (run / "samples" / sid / "scene_recipe.json").exists()), None)
        if film_sample:
            root = run / "samples" / film_sample
            recipe = read_json(root / "scene_recipe.json")
            film_object = next((obj for obj in recipe["objects"] if obj.get("calibration_target_effective_opacity") is not None), None)
            if film_object:
                alpha_path = root / "variants" / "independent" / "independent" / film_object["id"] / "alpha.exr"
                if alpha_path.exists():
                    layer_alpha = read_exr(alpha_path)[:, :, 0]
                    support = layer_alpha > .01
                    interior = np.asarray(Image.fromarray((support.astype(np.uint8) * 255), mode="L").filter(ImageFilter.MinFilter(5))) > 0
                    measured = float(np.median(layer_alpha[interior])) if interior.any() else 0.0
                    target = float(film_object["calibration_target_effective_opacity"])
                    error = abs(measured - target)
                    checks.append({"name": f"effective_film_opacity_{film_sample}_{film_object['id']}",
                                   "target": target, "observed_median": measured, "absolute_error": error,
                                   "pass": bool(error <= .10)})
                else:
                    errors.append(f"missing thin-film calibration layer {film_sample}/{film_object['id']}")
            else:
                errors.append(f"thin-film calibration target is not declared in {film_sample}")
            expected = None
            for object_id in recipe["diagnostics"]["independent_order"]:
                path = root / "variants" / "independent" / "independent" / object_id / "alpha.exr"
                if not path.exists():
                    errors.append(f"missing thin-film alpha-over layer {film_sample}/{object_id}")
                    expected = None
                    break
                layer = read_exr(path)[:, :, 0]
                expected = layer if expected is None else layer + (1.0 - layer) * expected
            joint = root / "joint" / "alpha.exr"
            if expected is not None and joint.exists():
                actual = read_exr(joint)[:, :, 0]
                residual = np.abs(actual - expected)
                checks.append({"name": f"alpha_over_{film_sample}", "mae": float(residual.mean()),
                               "p95": float(np.percentile(residual, 95)),
                               "pass": bool(residual.mean() <= .02 and np.percentile(residual, 95) <= .1)})
        else:
            errors.append("no rendered thin-film sample is available for effective-opacity calibration")
        errors.extend(c["name"] for c in checks if not c["pass"])
        report = {"schema_version": "calibration_expansion_v1", "status": "pass" if not errors else "fail",
                  "checks": checks, "errors": errors}
        write_json(run / "calibration.json", report)
        return report

    """Run the small numerical export calibration on an existing run.

    The acceptance run contains the same controlled cases needed to freeze the
    export convention: target-removed Core renders are empty, opaque contact
    targets have unit alpha in their support, and the woven veil's joint alpha
    must equal alpha-over of its independently rendered semantic layers.
    """
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    empty = run / "calibration" / "empty_core" / "alpha.exr"
    if empty.exists():
        a = read_exr(empty)[:, :, 0]
        checks.append({"name": "empty_core_background", "max_abs_alpha": float(np.max(np.abs(a))), "pass": bool(np.max(np.abs(a)) <= 1e-5)})
    else:
        errors.append("missing empty Core background calibration render")
    opaque = run / "samples" / "C01" / "joint" / "alpha.exr"
    if opaque.exists():
        a = read_exr(opaque)[:, :, 0]
        support = a > .99
        err = float(np.max(np.abs(a[support] - 1.0))) if support.any() else float("inf")
        checks.append({"name": "opaque_target_alpha", "support_fraction": float(support.mean()), "max_abs_error": err, "pass": bool(err <= .02)})
    else:
        errors.append("missing opaque calibration render")
    for sid in ("V01", "V02"):
        root = run / "samples" / sid
        recipe = read_json(root / "scene_recipe.json")
        order = recipe["diagnostics"]["independent_order"]
        expected: np.ndarray | None = None
        for object_id in order:
            path = root / "variants" / "independent" / "independent" / object_id / "alpha.exr"
            if not path.exists():
                errors.append(f"missing calibration layer {sid}/{object_id}")
                expected = None
                break
            layer = read_exr(path)[:, :, 0]
            expected = layer if expected is None else layer + (1.0 - layer) * expected
        joint_path = root / "joint" / "alpha.exr"
        if expected is not None and joint_path.exists():
            actual = read_exr(joint_path)[:, :, 0]
            residual = np.abs(actual - expected)
            checks.append({"name": f"alpha_over_{sid}", "mae": float(residual.mean()), "p95": float(np.percentile(residual, 95)), "pass": bool(residual.mean() <= .02 and np.percentile(residual, 95) <= .1)})
        elif expected is not None:
            errors.append(f"missing joint calibration render {sid}")
    errors.extend(c["name"] for c in checks if not c["pass"])
    report = {"schema_version": "calibration_v1", "status": "pass" if not errors else "fail", "checks": checks, "errors": errors}
    write_json(run / "calibration.json", report)
    return report
