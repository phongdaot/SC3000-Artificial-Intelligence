# this code isn't really "pythonic", since i'm used to cpp, but i tried...

from loader import load
from types import MappingProxyType
import heapq

coords, costs, distances, adjs = load()

def build_graphs():
    # coordinates are not relevant for this instance
    fw_adjs = {node: [] for node in adjs}
    rev_adjs = {node: [] for node in adjs}

    for node, neighbors in adjs.items():
        for neighbor in neighbors:
            edge_dist = distances[(node, neighbor)]
            edge_cost = costs[(node, neighbor)]
            fw_adjs[node].append((neighbor, edge_dist, edge_cost))
            rev_adjs[neighbor].append((node, edge_dist, edge_cost))
    
    return MappingProxyType(fw_adjs), MappingProxyType(rev_adjs)

# helper dijkstras
def reverse_dijkstra(start_node, adjs, edge_cost_func):
    dist = {node: float("inf") for node in adjs}
    dist[start_node] = 0
    heap = [(0, start_node)]

    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for neighbor, edge_dist, edge_cost in adjs[node]:
            new_dist = d + edge_cost_func(edge_dist, edge_cost)
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    
    return dist

# dijkstras to calculate extremes for heuristics
def dijkstra(start_node, target_node, adjs, edge_cost_func):
    dist = {node: float("inf") for node in adjs}
    par = {node: None for node in adjs}
    dist[start_node] = 0
    heap = [(0, start_node)]

    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for neighbor, edge_dist, edge_cost in adjs[node]:
            new_dist = d + edge_cost_func(edge_dist, edge_cost)
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
                par[neighbor] = node

    path = []
    cur = target_node
    while cur is not None:
        path.append(cur)
        cur = par[cur]
    path.reverse()
    total_dist = sum(distances[(u, v)] for u, v in zip(path, path[1:]))
    total_cost = sum(costs[(u, v)] for u, v in zip(path, path[1:]))

    return total_dist, total_cost

def run() -> None:
    fw_adjs, rev_adjs = build_graphs()

    # Instance constants
    start_node = 1
    target_node = 50
    cost_budget = 287932

    # Precompute heuristics
    min_cost_to_target = reverse_dijkstra(target_node, rev_adjs, lambda d, c: c)
    min_dist_to_target = reverse_dijkstra(target_node, rev_adjs, lambda d, c: d)

    # Calculate the 2 extremes for the heuristic (min cost path, min dist path)
    min_dist, min_dist_cost = dijkstra(start_node, target_node, fw_adjs, lambda d, c: d)
    min_cost_dist, min_cost = dijkstra(start_node, target_node, fw_adjs, lambda d, c: c)

    alpha = (min_cost_dist - min_dist) / (min_dist_cost - min_cost)

    # mixed weight heuristic. based on Lagrangian relaxation, more details in the report
    mixed_weight = reverse_dijkstra(target_node, rev_adjs, lambda d, c: d + alpha * c)

    def heuristic(node, cost_so_far):
        # combination of 3 heuristics:

        # 1. the wall: if the lowest cost from node still can't get us to target within budget, then it's a dead end
        p1 = 0 if cost_so_far + min_cost_to_target[node] <= cost_budget else float("inf")

        # 2. the minimum distance to target, ignoring cost
        p2 = min_dist_to_target[node]

        # 3. mixed weight heuristic
        p3 = mixed_weight[node] - alpha * (cost_budget - cost_so_far)

        return max(p1, p2, p3)
    
    # A* search, similar to the uninformed version, but now with a heuristic function to estimate each state

    # The following code is copied from the uninformed version, with minor modifications to accommodate the heuristic
    # label[node] is a list of non-dominated composite states at node
    label = {node: [] for node in adjs} 

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

    start_f = heuristic(start_node, 0)
    pq = [(start_f, 0, start_node, 0)] # (f, dist, node, cost)
    label[start_node].append((0, 0)) # (distance, cost)
    dist_map[(start_node, 0)] = 0

    optimal = None

    # Dijkstra's on composite states
    while pq:
        f, dist, node, cost = heapq.heappop(pq)

        if dist_map.get((node, cost), None) != dist:
            continue

        if node == target_node:
            optimal = (dist, cost)
            break

        for neighbor, edge_dist, edge_cost in fw_adjs[node]:
            new_cost = cost + edge_cost
            if new_cost > cost_budget:
                continue

            h_val = heuristic(neighbor, new_cost)
            if h_val == float("inf"):
                continue

            new_dist = dist + edge_dist

            if dominated(label[neighbor], new_dist, new_cost):
                continue

            add_state(label[neighbor], new_dist, new_cost)


            prev_dist = dist_map.get((neighbor, new_cost), float("inf"))
            if new_dist < prev_dist:
                dist_map[(neighbor, new_cost)] = new_dist
                par[(neighbor, new_cost)] = (node, cost)
                f_val = new_dist + h_val
                heapq.heappush(pq, (f_val, new_dist, neighbor, new_cost))

    if optimal is None:
        print("Part 3 - Informed Search")
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

    print("Part 3 - Informed Search")
    print(f"Shortest path: {' -> '.join(map(str, path))}")
    print(f"Shortest distance: {optimal[0]}")
    print(f"Total energy cost: {optimal[1]}")
    