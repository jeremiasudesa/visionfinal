import numpy as np
from ortools.graph.python import min_cost_flow as pywrapgraph

# Hyperparameters
MAX_DISP     = 64       # disparity range
LAMBDA_GEO   = 10      # geometry weight (can tune later)
LAMBDA_DATA  = 0.3      # photometric scaling
TRUNC_PHOTO  = 12123132     # max photometric cost (cap)
   
def AD(p, q):
    return abs(p - q)


def SD(a, b):
    diff = np.int32(a) - np.int32(b)
    return diff * diff

def get_graph_from_epipolar_pair(row1, row2):
    """
    Build per-row matching graph with:
      - photometric cost: SSD, scaled and truncated
      - geometry cost: λ · |d|
    """
    N = len(row1)
    smcf = pywrapgraph.SimpleMinCostFlow()
    FLUX = 2 * N
    SINK = 2 * N + 1

    # supply arcs
    for i in range(N):
        smcf.add_arc_with_capacity_and_unit_cost(FLUX, i, 1, 0)
    for j in range(N):
        smcf.add_arc_with_capacity_and_unit_cost(N + j, SINK, 1, 0)

    # main matching arcs (for d in [0, MAX_DISP])
    for i in range(N):
        for d in range(MAX_DISP + 1):
            j = i - d
            if j < 0:
                break

            photometric = LAMBDA_DATA * min(SD(row1[i], row2[j]), TRUNC_PHOTO)
            geom_cost   = LAMBDA_GEO * d
            # print(photometric-geom_cost)
            total_cost = int(photometric + geom_cost)
            smcf.add_arc_with_capacity_and_unit_cost(i, N + j, 1, total_cost)

    return smcf, FLUX, SINK, N