from get_image_pairs import get_image_pairs
from solve import solve_pair

def main():
    for pair in get_image_pairs():
        pred = solve_pair(pair[0], pair[1])
        # TODO: ver el ground truth y comparar


if __name__  == "__main__":
    main()