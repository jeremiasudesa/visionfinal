from pathlib import Path

def get_image_pairs():
    image_pairs = []
    for entry in Path("MiddEval3/testF").iterdir():
        if entry.is_dir():
            pair = []
            for sub_entry in entry.iterdir():
                if sub_entry.is_file() and sub_entry.suffix.lower() == '.png':
                    pair.append(sub_entry)
            image_pairs.append(pair)
    return image_pairs
