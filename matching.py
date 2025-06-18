from ortools.graph.python import min_cost_flow as pywrapgraph

def get_matching_from_graph(smcf, FLUX_CAPACITOR, SINK, N, OCCLUSION_COST = 9999999):
    FLOW  = N  
    SOURCE = SINK + 1                                    
    smcf.add_arc_with_capacity_and_unit_cost(SOURCE, FLUX_CAPACITOR, FLOW, 0)
    smcf.add_arc_with_capacity_and_unit_cost(SOURCE, SINK, FLOW, OCCLUSION_COST)    
    smcf.set_node_supply(SOURCE, FLOW)
    smcf.set_node_supply(SINK,  -FLOW)

    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError(f"Min-cost flow failed with status {status}")

    matching = []
    for arc_idx in range(smcf.num_arcs()):
        if smcf.flow(arc_idx) == 1:
            u = smcf.tail(arc_idx)
            v = smcf.head(arc_idx)
            if 0 <= u < N and N <= v < 2 * N:
                matching.append((u, v - N))

    return matching, smcf.optimal_cost()

