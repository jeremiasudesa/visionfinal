from get_image_pairs import get_image_pairs
from solve import solve_pair
import random

def main():
    pairs = get_image_pairs()[2:3]
    for pair in pairs:
        solve_pair(pair[0], pair[1], pair[2])


if __name__  == "__main__":
    main()