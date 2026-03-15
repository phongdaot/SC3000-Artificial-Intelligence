# this code isn't really "pythonic", since i'm used to cpp, but i tried...

from loader import load
from types import MappingProxyType
import heapq

coords, costs, distances, adjs = load()

def build_graphs():
    # coordinates are not relevant for this instance
    fw_adjs = {node: [] for node in adjs}

    for node, neighbors in adjs.items():
        for neighbor in neighbors:
            edge_dist = distances[(node, neighbor)]
            edge_cost = costs[(node, neighbor)]
            fw_adjs[node].append((neighbor, edge_dist, edge_cost))
    
    return MappingProxyType(fw_adjs)

def run() -> None:
    start_node = 1
    target_node = 50
    cost_budget = 287932

    fw_adjs = build_graphs() 
    
    # label[node] is a list of non-dominated composite states at node
    label = {node: [] for node in fw_adjs} 

    # some helper funcs related to labels
    def dominated(states, dist, cost):
        # returns True if (dist, cost) is dominated by any state in states
        for d, c in states:
            if d <= dist and c <= cost:
                return True
        return False
    
    # add state, and prune all dominated
    def add_state(states, dist, cost):
        states[:] = [(d, c) for d, c in states if not (d >= dist and c >= cost)]
        states.append((dist, cost))

    dist_map = {} # distance dict for all visited composite states (node, cost)
    par = {} # parent dict for path reconstruction. (node, cost) -> (prev_node, prev_cost)

    pq = [(0, start_node, 0)] # (distance, node, cost)
    label[start_node].append((0, 0)) # (distance, cost)
    dist_map[(start_node, 0)] = 0

    optimal = None

    # Dijkstra's on composite states
    while pq:
        dist, node, cost = heapq.heappop(pq)

        if dist_map.get((node, cost), None) != dist:
            continue

        if node == target_node:
            optimal = (dist, cost)
            break

        for neighbor, edge_dist, edge_cost in fw_adjs[node]:
            new_cost = cost + edge_cost

            if new_cost > cost_budget:
                continue

            new_dist = dist + edge_dist

            if dominated(label[neighbor], new_dist, new_cost):
                continue

            add_state(label[neighbor], new_dist, new_cost)

            prev_dist = dist_map.get((neighbor, new_cost), float("inf"))
            if new_dist < prev_dist:
                dist_map[(neighbor, new_cost)] = new_dist
                par[(neighbor, new_cost)] = (node, cost)
                heapq.heappush(pq, (new_dist, neighbor, new_cost))

    if optimal is None:
        print("Part 2 - Uninformed Search")
        print("No feasible path within the cost budget exists.")
        return
    
    # Path reconstruction
    path = []
    cur = (target_node, optimal[1])
    while cur in par:
        node, cost = cur
        path.append(node)
        cur = par[cur]
    path.append(start_node)
    path.reverse()

    print("Part 2 - Uninformed Search")
    print(f"Shortest path: {' -> '.join(map(str, path))}")
    print(f"Shortest distance: {optimal[0]}")
    print(f"Total energy cost: {optimal[1]}")