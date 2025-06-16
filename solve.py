from build_graph_from_pair import get_graph
import cv2

def solve_pair(img1, img2):
    m1, m2 = cv2.imread(str(img1)), cv2.imread(str(img2))
    if m1 is None or m2 is None:
        raise ValueError(f"No pude leer imagenes: {img1}, {img2}")
    print(get_graph(m1, m2))