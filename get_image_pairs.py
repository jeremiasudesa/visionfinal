from pathlib import Path

def get_image_pairs():
    image_pairs = []
    for left, right in zip(Path("training/image_2").iterdir(), Path("training/image_3").iterdir()):
            image_pairs.append((left, right))
    return image_pairs
