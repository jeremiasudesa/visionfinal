import numpy as np
from ortools.graph.python import min_cost_flow as pywrapgraph

MAX_DISP = 40
LAMBDA_GEO = 20
LAMBDA_DATA = 0.4
TRUNC_PHOTO = 12123132

def AD(p, q):
    return abs(p - q)

def SD(a, b):
    diff = np.int32(a) - np.int32(b)
    return int(np.sum(diff * diff))  # color-aware SSD

def get_graph_from_epipolar_pair(row1, row2):
    N = len(row1)
    smcf = pywrapgraph.SimpleMinCostFlow()
    FLUX = 2 * N
    SINK = 2 * N + 1

    for i in range(N):
        smcf.add_arc_with_capacity_and_unit_cost(FLUX, i, 1, 0)
    for j in range(N):
        smcf.add_arc_with_capacity_and_unit_cost(N + j, SINK, 1, 0)

    for i in range(N):
        for d in range(MAX_DISP + 1):
            j = i - d
            if j < 0:
                break

            photometric = LAMBDA_DATA * min(SD(row1[i], row2[j]), TRUNC_PHOTO)
            geom_cost = LAMBDA_GEO * d
            total_cost = int(photometric + geom_cost)
            smcf.add_arc_with_capacity_and_unit_cost(i, N + j, 1, total_cost)

    return smcf, FLUX, SINK, N
