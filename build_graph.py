import numpy as np
from ortools.graph.python import min_cost_flow as pywrapgraph

def SD(p, q):
    dp = np.int32(p) - np.int32(q)
    return dp * dp

    
def AD(p, q):
    return abs(p - q)

def get_graph_from_epipolar_pair(row1, row2):
    smcf = pywrapgraph.SimpleMinCostFlow()
    N = len(row1)
    FLUX_CAPACITOR = 2 * N
    SINK = 2 * N + 1

    for i in range(N):
        smcf.add_arc_with_capacity_and_unit_cost(FLUX_CAPACITOR, i, 1, 0)

    for j in range(N):
        smcf.add_arc_with_capacity_and_unit_cost(N + j, SINK, 1, 0)

    for i in range(N):
        for j in range(N):
            cost = SD(row1[i], row2[j])
            smcf.add_arc_with_capacity_and_unit_cost(i, N + j, 1, cost)

    return smcf, FLUX_CAPACITOR, SINK, N
