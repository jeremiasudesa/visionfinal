import numpy as np
import cv2
import os
from build_graph import get_graph_from_epipolar_pair
from matching import get_matching_from_graph

SCALE = 10

def path_to_img(path):
    m = cv2.imread(str(path))
    if m is None:
        raise ValueError(f"No pude leer la imagen: {path}")
    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return cv2.resize(m, (len(m[0])//SCALE, len(m)//SCALE), interpolation=cv2.INTER_AREA)

def solve_pair(img1, img2):
    orig_left = cv2.imread(str(img1), cv2.IMREAD_GRAYSCALE)
    if orig_left is None:
        raise ValueError(f"No pude leer la imagen original: {img1}")
    orig_h, orig_w = orig_left.shape

    left  = path_to_img(img1)
    right = path_to_img(img2)
    h, w = left.shape

    disp = np.zeros((h, w), dtype=np.float32)
    for row_idx, (row_l, row_r) in enumerate(zip(left, right)):
        solver, F, S, N = get_graph_from_epipolar_pair(row_l, row_r)
        matches, _ = get_matching_from_graph(solver, F, S, N)
        for u, v in matches:
            disp[row_idx, u] = u - v
        if row_idx % 10 == 0:
            os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{row_idx+1}/{h}")

    disp_norm = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_norm = np.uint8(disp_norm)

    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_BONE)

    left_color_full = cv2.cvtColor(orig_left, cv2.COLOR_GRAY2BGR)
    left_color_full = cv2.resize(left_color_full, (orig_w//SCALE, orig_h//SCALE), interpolation=cv2.INTER_LINEAR)

    combined = np.hstack((left_color_full, disp_color))

    cv2.imshow("Left Original | Disparity Heatmap (upscaled)", combined)
    try:
        while True:
            if cv2.getWindowProperty("Left Original | Disparity Heatmap (upscaled)", cv2.WND_PROP_VISIBLE) < 1:
                break
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    return disp, disp_norm
