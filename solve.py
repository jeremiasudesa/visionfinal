#!/usr/bin/env python3
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from build_graph import get_graph_from_epipolar_pair
from matching import get_matching_from_graph

# -------------------------------------------------------------
# Parameters (tweak for speed / quality)
# -------------------------------------------------------------
SCALE    = 2     # down-sample factor
ROW_STEP = 3     # only process every 5th row in CPU mode

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def write_pfm(path: str, image: np.ndarray, scale: float = 1.0):
    """
    Write a float32 single-channel or three-channel image to a PFM file.
    """
    if image.dtype.name != "float32":
        raise ValueError("PFM requires float32 data")
    image = np.flipud(image)
    color = (image.ndim == 3 and image.shape[2] == 3)
    with open(path, "wb") as f:
        f.write(b"PF\n" if color else b"Pf\n")
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode())
        endian = image.dtype.byteorder
        if endian == "<" or (endian == "=" and np.little_endian):
            scale = -scale
        f.write(f"{scale}\n".encode())
        image.tofile(f)


def process_row(row_idx: int, left_row: np.ndarray, right_row: np.ndarray):
    """
    Build the epipolar graph for one scanline and solve for matches.
    """
    solver, F, S, N = get_graph_from_epipolar_pair(left_row, right_row)
    matches, _      = get_matching_from_graph(solver, F, S, N)
    return row_idx, matches

# -------------------------------------------------------------
# Main stereo + depth function (no inpainting)
# -------------------------------------------------------------
def solve_pair(
    img1: Path,
    img2: Path,
    calib_file: Path = None,
    use_cuda: bool = True
):
    """
    Compute disparity (and optional metric depth) for a stereo pair.

    Args:
      img1, img2: Path to left/right BGR images.
      calib_file: optional Path to KITTI raw-calib .txt.
      use_cuda: if True and a CUDA device is available, uses cv2.cuda StereoSGM.

    Returns:
      disp:      float32 disparity map (downsampled)
      disp_norm: uint8 0–255 normalized disparity
      disp_color:uint8 3-channel color-mapped disparity
    """
    # Load
    left  = cv2.imread(str(img1), cv2.IMREAD_COLOR)
    right = cv2.imread(str(img2), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise FileNotFoundError(f"Cannot read {img1} or {img2}")

    h0, w0 = left.shape[:2]
    # Downsample
    left  = cv2.resize(left,  (w0//SCALE, h0//SCALE), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (w0//SCALE, h0//SCALE), interpolation=cv2.INTER_AREA)
    h, w  = left.shape[:2]

    # Buffers
    disp         = np.zeros((h, w), dtype=np.float32)
    been_matched = np.zeros((h, w), dtype=bool)

    # Disparity estimation
    if use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        gpuL = cv2.cuda_GpuMat(); gpuR = cv2.cuda_GpuMat()
        gpuL.upload(left); gpuR.upload(right)
        min_disp, num_disp = 0, 128
        sgm = cv2.cuda.createStereoSGM(minDisp=min_disp, numDisparities=num_disp)
        sgm.setP1(8 * 3 * 3 * 3); sgm.setP2(32 * 3 * 3 * 3)
        gpu_disp = sgm.compute(gpuL, gpuR)
        disp16   = gpu_disp.download()
        disp[:]  = disp16.astype(np.float32) / 16.0
        been_matched[:] = disp16 > (min_disp * 16)
    else:
        rows = [(i, left[i], right[i]) for i in range(0, h, ROW_STEP)]
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_row, i, rl, rr) for (i, rl, rr) in rows]
            for future in as_completed(futures):
                row_idx, matches = future.result()
                for u, v in matches:
                    disp[row_idx, u] = float(u - v)
                    been_matched[row_idx, u] = True

    # Normalize & color-map
    disp_norm  = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
    disp_color[~been_matched] = (0, 0, 0)

    # Optional depth conversion
    if calib_file is not None:
        calib = {}
        with open(calib_file, "r") as f:
            for line in f:
                line = line.strip()
                if ":" not in line:
                    continue
                key, vals = line.split(":", 1)
                data = np.fromstring(vals, sep=" ")
                calib[key] = data
        # get P_rect_02 or fallback
        if "P_rect_02" in calib:
            P = calib["P_rect_02"].reshape(3, 4)
        else:
            pkeys = [k for k in calib if k.startswith("P")]
            if not pkeys:
                raise KeyError(f"No P_* found in {calib_file}")
            P = calib[pkeys[0]].reshape(3, 4)
        fx       = float(P[0, 0])
        baseline = -float(P[0, 3]) / fx
        depth = np.zeros_like(disp, dtype=np.float32)
        valid = disp > 0
        depth[valid] = (fx * baseline) / disp[valid]
        write_pfm("depth.pfm", depth)

    # Save outputs
    cv2.imwrite("disparity.png", disp_norm)
    cv2.imwrite("disparity_color.jpg", disp_color)

    # Display results
    vis_left  = cv2.resize(left,  (w, h), interpolation=cv2.INTER_LINEAR)
    vis_right = cv2.resize(right, (w, h), interpolation=cv2.INTER_LINEAR)
    montage   = np.vstack([vis_left, disp_color, vis_right])
    cv2.imshow("Left | Disp | Right", montage)
    try:
        while cv2.getWindowProperty("Left | Disp | Right", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    return disp, disp_norm, disp_color
