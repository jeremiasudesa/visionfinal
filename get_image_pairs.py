from pathlib import Path
from typing import List, Tuple

def get_image_pairs() -> List[Tuple[Path, Path, Path]]:
    """
    Scan the KITTI folders under `training/` and return triplets:
      (image_2/*_10.png, image_3/*_11.png, calib_cam_to_cam/*.txt)
    matched by the six‐digit frame ID.
    """
    base      = Path("training")
    img2_dir  = base / "image_2"
    img3_dir  = base / "image_3"
    calib_dir = base / "calib_cam_to_cam"

    # collect left images ending in _10.png
    left_map = {}
    for p in img2_dir.iterdir():
        if p.suffix.lower() != ".png":
            continue
        parts = p.stem.split("_")
        if len(parts) == 2 and parts[1] == "10":
            frame_id = parts[0]
            left_map[frame_id] = p

    # collect right images ending in _11.png
    right_map = {}
    for p in img3_dir.iterdir():
        if p.suffix.lower() != ".png":
            continue
        parts = p.stem.split("_")
        if len(parts) == 2 and parts[1] == "11":
            frame_id = parts[0]
            right_map[frame_id] = p

    # collect calibration files named frame_id.txt
    calib_map = {}
    for p in calib_dir.iterdir():
        if p.suffix.lower() == ".txt":
            frame_id = p.stem
            calib_map[frame_id] = p

    # intersect frame IDs
    common_ids = sorted(left_map.keys() & right_map.keys() & calib_map.keys())

    # build triplets
    triplets = [(left_map[f], right_map[f], calib_map[f]) for f in common_ids]
    return triplets

