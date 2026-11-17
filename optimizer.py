import heapq
import math
import numpy as np

GRID_W, GRID_H = 20, 15

def astar(start, goal, obstacles):
    """4-directional A* on a grid. Returns a list of grid cells or [] if blocked."""
    start, goal = tuple(start), tuple(goal)
    blocked = {tuple(p) for p in obstacles}
    if start in blocked or goal in blocked:
        return []
    pq = [(0, 0, start)]
    came = {}
    g = {start: 0}
    counter = 0
    while pq:
        _, cost, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if cost != g.get(cur):
            continue
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nxt = (cur[0]+dx, cur[1]+dy)
            if not (0 <= nxt[0] < GRID_W and 0 <= nxt[1] < GRID_H) or nxt in blocked:
                continue
            ng = cost + 1
            if ng < g.get(nxt, 10**9):
                came[nxt] = cur
                g[nxt] = ng
                counter += 1
                h = abs(goal[0]-nxt[0]) + abs(goal[1]-nxt[1])
                heapq.heappush(pq, (ng+h, ng, nxt))
    return []

def route_for_uav(uav, task_ids, tasks, obstacles):
    current = tuple(uav["start"])
    full_path = [current]
    task_arrivals = []
    distance = 0.0
    elapsed = 0.0
    energy = 0.0
    for tid in task_ids:
        task = tasks[tid]
        segment = astar(current, task["pos"], obstacles)
        if not segment:
            return {"path": full_path, "distance": 1e6, "energy": 1e6,
                    "time": 1e6, "arrivals": [], "feasible": False}
        seg_dist = max(0, len(segment)-1)
        distance += seg_dist
        # Speed is in grid-cells per time unit; energy_per_cell is a simple model parameter.
        seg_time = seg_dist / max(0.1, float(uav["speed"]))
        elapsed += seg_time
        energy += seg_dist * float(uav.get("energy_per_cell", 1.0))
        task_arrivals.append((tid, elapsed))
        full_path.extend(segment[1:])
        current = tuple(task["pos"])
    return {"path": full_path, "distance": distance, "energy": energy,
            "time": elapsed, "arrivals": task_arrivals, "feasible": True}

def decode(position, uavs, tasks):
    n_uav = len(uavs)
    n_task = len(tasks)
    assign_keys = position[:n_task]
    order_keys = position[n_task:]
    allocation = [[] for _ in uavs]
    for tid in range(n_task):
        uid = min(n_uav-1, int(assign_keys[tid] * n_uav))
        allocation[uid].append(tid)
    ordered = []
    for uid, ids in enumerate(allocation):
        ordered.append(sorted(ids, key=lambda tid: order_keys[tid]))
    return ordered

def evaluate(position, uavs, tasks, obstacles):
    allocation = decode(position, uavs, tasks)
    routes = []
    total_distance = total_energy = 0.0
    completion_times = []
    workload = []
    priority_delay = 0.0
    battery_penalty = 0.0
    infeasible_penalty = 0.0

    for uid, ids in enumerate(allocation):
        route = route_for_uav(uavs[uid], ids, tasks, obstacles)
        routes.append(route)
        total_distance += route["distance"]
        total_energy += route["energy"]
        completion_times.append(route["time"])
        workload.append(route["time"])
        if not route["feasible"]:
            infeasible_penalty += 5000
        if route["energy"] > float(uavs[uid]["battery"]):
            battery_penalty += (route["energy"] - float(uavs[uid]["battery"])) * 80
        for tid, arrival in route["arrivals"]:
            # High-priority tasks should ideally be reached earlier.
            priority_delay += arrival * float(tasks[tid]["priority"])

    # Estimate simultaneous cell occupancy. A collision-risk penalty is applied
    # when two UAV paths occupy the same cell at the same discrete time step.
    collision_count = 0
    max_len = max((len(r["path"]) for r in routes), default=0)
    for t in range(max_len):
        occupied = []
        for r in routes:
            if r["path"]:
                occupied.append(r["path"][min(t, len(r["path"])-1)])
        collision_count += len(occupied) - len(set(occupied))

    if workload:
        imbalance = float(np.std(workload))
        makespan = max(workload)
    else:
        imbalance = makespan = 0.0

    # Weighted multi-objective fitness. Lower is better.
    fitness = (
        1.0 * total_distance +
        1.5 * total_energy +
        3.0 * makespan +
        2.0 * imbalance +
        0.8 * priority_delay +
        120.0 * collision_count +
        battery_penalty +
        infeasible_penalty
    )
    return {
        "fitness": float(fitness), "allocation": allocation, "routes": routes,
        "distance": float(total_distance), "energy": float(total_energy),
        "makespan": float(makespan), "imbalance": float(imbalance),
        "collisions": int(collision_count), "battery_penalty": float(battery_penalty),
        "priority_delay": float(priority_delay)
    }

def run_pso(uavs, tasks, obstacles, particles=36, iterations=65, seed=7):
    """Continuous PSO keys are decoded into discrete UAV assignments and task order."""
    rng = np.random.default_rng(seed)
    n = len(tasks)
    dim = 2*n
    x = rng.random((particles, dim))
    v = rng.uniform(-0.08, 0.08, (particles, dim))
    pbest = x.copy()
    pbest_scores = np.array([evaluate(p, uavs, tasks, obstacles)["fitness"] for p in x])
    gi = int(np.argmin(pbest_scores))
    gbest = pbest[gi].copy()
    gbest_score = float(pbest_scores[gi])
    history = [gbest_score]

    for it in range(iterations):
        w = 0.72 - 0.32 * (it / max(1, iterations-1))
        r1, r2 = rng.random((2, particles, dim))
        v = w*v + 1.45*r1*(pbest-x) + 1.45*r2*(gbest-x)
        v = np.clip(v, -0.25, 0.25)
        x = np.clip(x+v, 0.0, 0.999999)
        scores = np.array([evaluate(p, uavs, tasks, obstacles)["fitness"] for p in x])
        improved = scores < pbest_scores
        pbest[improved] = x[improved]
        pbest_scores[improved] = scores[improved]
        gi = int(np.argmin(pbest_scores))
        if pbest_scores[gi] < gbest_score:
            gbest_score = float(pbest_scores[gi])
            gbest = pbest[gi].copy()
        history.append(gbest_score)

    result = evaluate(gbest, uavs, tasks, obstacles)
    result["history"] = history
    result["iterations"] = iterations
    result["particles"] = particles
    return result
