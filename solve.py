"""
Stereo → disparity → metric depth (KITTI-style calibration)

∙ Shows a montage (Left | Disparity | Right) in an OpenCV window.
∙ When you close that window, pops-up a Matplotlib figure of depth
  with a colour bar labelled in metres.
∙ Saves:
      disparity.png         - single-channel 8-bit disparity
      disparity_color.jpg   - colour-mapped disparity
      depth.pfm             - 32-bit floating-point depth (metres)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------------------------------------------------
# Parameters you might want to tweak
# -------------------------------------------------------------
SCALE    = 2      # 1 = full res, 2 = half res, 4 = quarter res …
ROW_STEP = 2      # evaluate every n-th scanline to speed-up CPU graph solver

# -------------------------------------------------------------
# Stereo graph / matching stubs – keep your own implementations
# -------------------------------------------------------------
from build_graph import get_graph_from_epipolar_pair
from matching    import get_matching_from_graph


# -------------------------------------------------------------
# Utility: write a float32 image to PFM (KITTI’s native depth format)
# -------------------------------------------------------------
def write_pfm(path: str, image: np.ndarray, scale: float = 1.0) -> None:
    if image.dtype != np.float32:
        raise ValueError("PFM requires float32 data")
    image = np.flipud(image)
    color = image.ndim == 3 and image.shape[-1] == 3
    with open(path, "wb") as f:
        f.write(b"PF\n" if color else b"Pf\n")
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode())
        endian = image.dtype.byteorder
        if endian == "<" or (endian == "=" and np.little_endian):
            scale = -scale
        f.write(f"{scale}\n".encode())
        image.tofile(f)


# -------------------------------------------------------------
# Parse KITTI calibration and return (fx_px, baseline_m)
# -------------------------------------------------------------
def parse_kitti_calib(calib_path: Path) -> tuple[float, float]:
    calib = {}
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            k, vals = line.split(":", 1)
            calib[k.strip()] = np.fromstring(vals, sep=" ")

    # Prefer rectified projection matrices if present; fall back otherwise
    p_left_key  = "P_rect_02" if "P_rect_02" in calib else "P2"
    p_right_key = "P_rect_03" if "P_rect_03" in calib else "P3"

    if p_left_key not in calib or p_right_key not in calib:
        raise KeyError(f"Need {p_left_key} and {p_right_key} in {calib_path}")

    P_left  = calib[p_left_key ].reshape(3, 4)
    P_right = calib[p_right_key].reshape(3, 4)

    fx_px       = float(P_left[0, 0])
    cx_left_m   = -float(P_left [0, 3]) / fx_px    # Tx/fx in metres
    cx_right_m  = -float(P_right[0, 3]) / fx_px
    baseline_m  = abs(cx_right_m - cx_left_m)      # > 0

    return fx_px, baseline_m


# -------------------------------------------------------------
# Worker for one scanline (graph construction + matching)
# -------------------------------------------------------------
def _solve_row(row_idx: int, left_row: np.ndarray, right_row: np.ndarray):
    solver, F, S, N = get_graph_from_epipolar_pair(left_row, right_row)
    matches, _      = get_matching_from_graph(solver, F, S, N)
    return row_idx, matches


# -------------------------------------------------------------
# Main function
# -------------------------------------------------------------
def solve_pair(
    img1: Path,
    img2: Path,
    calib_file: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(img1)
    # ---------- I/O ----------
    left  = cv2.imread(str(img1), cv2.IMREAD_COLOR)
    right = cv2.imread(str(img2), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise FileNotFoundError(f"Cannot read {img1} or {img2}")

    h0, w0 = left.shape[:2]
    left  = cv2.resize(left,  (w0 // SCALE, h0 // SCALE), cv2.INTER_AREA)
    right = cv2.resize(right, (w0 // SCALE, h0 // SCALE), cv2.INTER_AREA)
    h, w  = left.shape[:2]

    # ---------- Disparity buffer ----------
    disp         = np.zeros((h, w), dtype=np.float32)
    been_matched = np.zeros((h, w), dtype=bool)

    # ---------- Solve every ROW_STEP-th line in parallel ----------
    rows = [(i, left[i], right[i]) for i in range(0, h, ROW_STEP)]
    with ProcessPoolExecutor() as pool:
        futures = [pool.submit(_solve_row, i, rl, rr) for (i, rl, rr) in rows]
        for fut in as_completed(futures):
            row_idx, matches = fut.result()
            for u, v in matches:
                disp[row_idx, u] = float(u - v)
                been_matched[row_idx, u] = True

    # ---------- Visualise disparity ----------
    disp_norm  = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
    disp_color[~been_matched] = (0, 0, 0)

    # ---------- Optional metric depth ----------
    depth = None
    if calib_file is not None:
        fx_px, baseline_m = parse_kitti_calib(Path(calib_file))
        fx_scaled         = fx_px / SCALE                   # account for down-sampling
        depth             = np.zeros_like(disp, dtype=np.float32)
        valid             = disp > 0
        depth[valid]      = (fx_scaled * baseline_m) / disp[valid]
        write_pfm("depth.pfm", depth)

    # ---------- Save quick-look PNGs ----------
    cv2.imwrite("disparity.png",      disp_norm)
    cv2.imwrite("disparity_color.jpg", disp_color)

    # ---------- First window: three-way montage ----------
    montage = np.vstack([
        left,                     # top
        disp_color,               # middle
        right                     # bottom
    ])
    cv2.imshow("Left | Disparity | Right", montage)
    try:
        while cv2.getWindowProperty("Left | Disparity | Right",
                                    cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(1) in (27, ord('q')):
                break
    finally:
        cv2.destroyAllWindows()

    # ---------- Second window: depth with colour bar (OpenCV) ----------
    if depth is not None:
        # Normalize depth to 0–255 for visualization
        depth_vis = depth.copy()
        depth_vis[~np.isfinite(depth_vis)] = 0  # remove inf/nan
        norm_depth = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
        norm_depth = norm_depth.astype(np.uint8)

        # Apply a colormap
        depth_color = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)

        # Create a vertical color bar
        bar_height = depth_color.shape[0]
        bar_width = 40
        gradient = np.linspace(255, 0, bar_height, dtype=np.uint8).reshape(-1, 1)
        colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        colorbar = cv2.resize(colorbar, (bar_width, bar_height), interpolation=cv2.INTER_NEAREST)

        # Annotate color bar with labels (e.g., min/max depth)
        min_depth = np.min(depth[depth > 0])
        max_depth = np.max(depth)
        colorbar_annotated = colorbar.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(colorbar_annotated, f"{max_depth:.1f}m", (2, 15), font, 0.4, (255, 255, 255), 1)
        cv2.putText(colorbar_annotated, f"{min_depth:.1f}m", (2, bar_height - 5), font, 0.4, (255, 255, 255), 1)

        depth_color[~been_matched] = (0, 0, 0)
        # Concatenate depth map and color bar
        depth_display = np.hstack((depth_color, colorbar_annotated))

        cv2.imshow("Metric Depth (OpenCV)", depth_display)
        try:
            while cv2.getWindowProperty("Metric Depth (OpenCV)", cv2.WND_PROP_VISIBLE) >= 1:
                if cv2.waitKey(1) in (27, ord('q')):
                    break
        finally:
            cv2.destroyAllWindows()

    return disp, disp_norm, disp_color


# -------------------------------------------------------------
# Example CLI usage
# -------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Stereo → disparity → metric depth with visualisation")
    ap.add_argument("left",             type=Path, help="left image (BGR)")
    ap.add_argument("right",            type=Path, help="right image (BGR)")
    ap.add_argument("--calib", "-c",    type=Path, help="KITTI raw-calib *.txt")
    args = ap.parse_args()

    solve_pair(args.left, args.right, args.calib)
