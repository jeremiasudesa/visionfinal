import numpy as np
import os
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def resize_and_rescale_gt(gt_disp, shape, scale):
    from scipy.ndimage import zoom
    zoom_factors = (shape[0] / gt_disp.shape[0], shape[1] / gt_disp.shape[1])
    gt_resized = zoom(gt_disp, zoom_factors, order=1)
    return gt_resized / scale  # rescale disparities

def compute_metrics(pred_path, gt_path, mask_threshold=1.0):
    pred, _ = read_pfm(pred_path)
    gt, _   = read_pfm(gt_path)

    if pred.shape != gt.shape:
        print(f"Resizing and rescaling GT from {gt.shape} to {pred.shape}")
        gt = resize_and_rescale_gt(gt, pred.shape, scale=4)  # your SCALE value

    valid = (gt > 0) & (gt < 1e4)
    error = np.abs(pred[valid] - gt[valid])

    bad2 = np.mean(error > 2) * 100
    mae  = np.mean(error)
    rms  = np.sqrt(np.mean(error**2))

    print(f"Bad-2: {bad2:.2f}%  |  MAE: {mae:.2f} px  |  RMS: {rms:.2f} px")

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axs[0].imshow(pred, cmap='plasma')
    axs[0].set_title("Predicted Disparity")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(gt, cmap='plasma')
    axs[1].set_title("Ground Truth Disparity")
    plt.colorbar(im1, ax=axs[1])

    plt.suptitle("Disparity Maps")
    plt.tight_layout()
    plt.show()

# Example:

def read_pfm(file):
    import re
    import numpy as np

    with open(file, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise ValueError("Not a PFM file.")

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if not dim_match:
            raise ValueError("Malformed PFM header.")
        width, height = map(int, dim_match.groups())

        scale = float(f.readline().decode('utf-8').strip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f')

        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # flip vertically
        return data, scale


compute_metrics("MiddEval3/trainingF/Adirondack/pred_disp.pfm", "MiddEval3/trainingF/Adirondack/disp0GT.pfm")
