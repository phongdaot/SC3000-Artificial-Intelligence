# this code isn't really "pythonic", since i'm used to cpp, but i tried...

from loader import load
from types import MappingProxyType
import heapq

coords, costs, distances, adjs = load()

def build_graphs():
    # coordinates and costs are not relevant for this relaxed instance
    fw_adjs = {node: [] for node in adjs}
    rev_adjs = {node: [] for node in adjs}

    for node, neighbors in adjs.items():
        for neighbor in neighbors:
            edge_dist = distances[(node, neighbor)]
            fw_adjs[node].append((neighbor, edge_dist))
            rev_adjs[neighbor].append((node, edge_dist))
    
    return MappingProxyType(fw_adjs), MappingProxyType(rev_adjs)

def run() -> None:
    fw_adjs, rev_adjs = build_graphs()

    # Bidirectional Dijkstra's
    # We alternate between forward and backward search, relaxing 1 node at a time in each direction.

    # Variable setups
    start_node = 1
    target_node = 50

    state = {"cur_fw": start_node, "cur_bk": target_node, "best": float("inf"), "meet_edge": None}

    # current distance, used in dijkstra's. not to be confused with edge distances stored in fw_adjs/rev_adjs.
    fw_dist = {node: float("inf") for node in fw_adjs}
    bk_dist = {node: float("inf") for node in fw_adjs}

    fw_dist[start_node] = 0
    bk_dist[target_node] = 0

    # dijkstra heaps
    fw_heap = [(0, start_node)]
    bk_heap = [(0, target_node)]

    # for path tracing
    par_fw = {node: None for node in fw_adjs}
    par_bk = {node: None for node in fw_adjs}

    # indicates whether a node has been relaxed
    closed_fw = {node: False for node in fw_adjs}
    closed_bk = {node: False for node in fw_adjs}

    # Relax one node
    def relax_forward() -> bool: # returns False if no relation can/should be done, True otherwise
        while fw_heap:
            d, node = fw_heap[0]
            if d > fw_dist[node]: 
                heapq.heappop(fw_heap)  # Outdated entry, skip
                continue
            break

        if not fw_heap: return False  # No more nodes to relax

        d, node = heapq.heappop(fw_heap)
        state["cur_fw"] = node
        closed_fw[node] = True

        if d + bk_dist[state["cur_bk"]] >= state["best"]:
            return False # we can just prune here
        
        for neighbor, edge_dist in fw_adjs[node]:
            if not closed_fw[neighbor]:
                new_dist = d + edge_dist
                if new_dist < fw_dist[neighbor]:
                    fw_dist[neighbor] = new_dist
                    par_fw[neighbor] = node
                    heapq.heappush(fw_heap, (new_dist, neighbor))
            
            if closed_bk[neighbor]:
                # this edge is a potential meeting point
                meet_dist = d + edge_dist + bk_dist[neighbor]
                if meet_dist < state["best"]:
                    state["best"] = meet_dist
                    state["meet_edge"] = (node, neighbor)

        return True

    def relax_backward() -> bool:
        while bk_heap:
            d, node = bk_heap[0]
            if d > bk_dist[node]: 
                heapq.heappop(bk_heap)  # Outdated entry, skip
                continue
            break

        if not bk_heap: return False  # No more nodes to relax

        d, node = heapq.heappop(bk_heap)
        state["cur_bk"] = node
        closed_bk[node] = True

        if d + fw_dist[state["cur_fw"]] >= state["best"]:
            return False # No need to relax further, we won't find a better path
        
        for neighbor, edge_dist in rev_adjs[node]:
            if not closed_bk[neighbor]:
                new_dist = d + edge_dist
                if new_dist < bk_dist[neighbor]:
                    bk_dist[neighbor] = new_dist
                    par_bk[neighbor] = node
                    heapq.heappush(bk_heap, (new_dist, neighbor))
            
            if closed_fw[neighbor]:
                # this edge is a potential meeting point
                meet_dist = d + edge_dist + fw_dist[neighbor]
                if meet_dist < state["best"]:
                    state["best"] = meet_dist
                    state["meet_edge"] = (neighbor, node) # we reverse this to keep it consistent: first node is forward, second node is backward
        
        return True
    
    # Main loop
    turn = 1
    while True:
        if turn:
            if not relax_forward():
                break
        else:
            if not relax_backward():
                break
        
        turn ^= 1 # alternate

    # Path reconstruction
    fw_path = []
    end_fw = state["meet_edge"][0]
    while end_fw is not None:
        fw_path.append(end_fw)
        end_fw = par_fw[end_fw]

    bk_path = []
    end_bk = state["meet_edge"][1]
    while end_bk is not None:
        bk_path.append(end_bk)
        end_bk = par_bk[end_bk]

    fw_path.reverse()
    full_path = fw_path + bk_path

    # Print output
    print("Part 1 - Relaxed instance")
    print(f"Shortest path: {' -> '.join(map(str, full_path))}")
    print(f"Shortest distance: {state['best']}")
    print(f"Total energy cost: {sum(costs[(u,v)] for u, v in zip(full_path, full_path[1:]))}")
    