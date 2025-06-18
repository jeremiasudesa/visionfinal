import numpy as np
import cv2
import os
from build_graph import get_graph_from_epipolar_pair
from matching import get_matching_from_graph
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools


def segment_based_smooth_and_fill(disp, left_gray, been_matched,
                                  region_size=35, ruler=20.0,
                                  min_valid=6, show=False):
    """
    Smooth and fill disparity using SLIC segmentation and guided filtering.

    - Step 1: For each SLIC segment, replace all pixels with median of valid disparities
    - Step 2: Apply guided filtering to fill remaining holes using left image as guidance

    Parameters
    ----------
    disp : float32 disparity map (may contain zeros)
    left_gray : grayscale left image (same size as disp)
    been_matched : boolean mask, True where disparity is valid
    region_size : SLIC superpixel size
    ruler : SLIC compactness (higher = smoother blobs)
    min_valid : minimum number of valid disparities to fill a segment
    show : if True, display SLIC boundaries

    Returns
    -------
    filled_disp : float32, fully filled disparity map
    """

    img_bgr = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)

    slic = cv2.ximgproc.createSuperpixelSLIC(
        img_bgr,
        algorithm=cv2.ximgproc.SLIC,
        region_size=region_size,
        ruler=ruler
    )
    slic.iterate(10)
    labels = slic.getLabels()
    n_labels = slic.getNumberOfSuperpixels()

    smoothed = disp.copy()

    for lbl in range(n_labels):
        mask = (labels == lbl)
        valid_vals = disp[mask & been_matched]
        if valid_vals.size >= min_valid:
            smoothed[mask] = np.median(valid_vals)

    # Optional: show segmentation overlay
    if show:
        contour = slic.getLabelContourMask(thick_line=True)
        vis = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
        vis[contour == 255] = (0, 255, 0)
        cv2.imshow("SLIC boundaries", vis)
        cv2.waitKey(1)

    # Fill remaining holes using guided filtering
    filled_disp = cv2.ximgproc.guidedFilter(
        guide=left_gray,
        src=smoothed.astype(np.float32),
        radius=9,
        eps=1e-2
    )

    return filled_disp


SCALE = 4

def path_to_img(path):
    m = cv2.imread(str(path))
    if m is None:
        raise ValueError(f"No pude leer la imagen: {path}")
    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = cv2.resize(m, (len(m[0])//SCALE, len(m)//SCALE), interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur(m, (5, 5), 0)
    # cv2.imshow("Left Original | Disparity Heatmap (upscaled)", m)
    # try:
    #     while True:
    #         if cv2.getWindowProperty("Left Original | Disparity Heatmap (upscaled)", cv2.WND_PROP_VISIBLE) < 1:
    #             break
    #         cv2.waitKey(1)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     cv2.destroyAllWindows()

    return m

def process_row(row_idx, row_l, row_r):
    solver, F, S, N = get_graph_from_epipolar_pair(row_l, row_r)
    matches, _ = get_matching_from_graph(solver, F, S, N)
    return row_idx, matches

def solve_pair(img1, img2):
    orig_left = cv2.imread(str(img1), cv2.IMREAD_GRAYSCALE)
    orig_right = cv2.imread(str(img2), cv2.IMREAD_GRAYSCALE)
    if orig_left is None or orig_right is None:
        raise ValueError(f"No pude leer una de las imágenes: {img1}, {img2}")
    orig_h, orig_w = orig_left.shape

    left = path_to_img(img1)
    right = path_to_img(img2)
    h, w = left.shape

    disp = np.zeros((h, w), dtype=np.float32)
    been_matched = np.zeros((h, w), dtype=bool)

    rows = [(idx, left[idx], right[idx]) for idx in range(h)]

    print("Processing with parallelism...")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_row, idx, row_l, row_r) for idx, row_l, row_r in rows]
        for count, future in enumerate(as_completed(futures), 1):
            row_idx, matches = future.result()
            for u, v in matches:
                disp[row_idx, u] = u - v
                been_matched[row_idx, u] = True
            if count % 10 == 0:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"{count}/{h}")

    final_disp = segment_based_smooth_and_fill(disp, left, been_matched, show=True)

    disp_norm = cv2.normalize(final_disp, None, 0, 255,
                            norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_BONE)

    # Paint unmatched (occluded) pixels red in the color image
    red = np.array([0, 0, 255], dtype=np.uint8)  # BGR
    disp_color[~been_matched] = red
    left_color = cv2.cvtColor(orig_left, cv2.COLOR_GRAY2BGR)
    right_color = cv2.cvtColor(orig_right, cv2.COLOR_GRAY2BGR)

    left_color = cv2.resize(left_color, (orig_w // SCALE, orig_h // SCALE), interpolation=cv2.INTER_LINEAR)
    right_color = cv2.resize(right_color, (orig_w // SCALE, orig_h // SCALE), interpolation=cv2.INTER_LINEAR)

    combined = np.hstack((left_color, disp_color, right_color))

    cv2.imshow("Left | Disparity | Right", combined)
    try:
        while True:
            if cv2.getWindowProperty("Left | Disparity | Right", cv2.WND_PROP_VISIBLE) < 1:
                break
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    return disp, disp_norm
