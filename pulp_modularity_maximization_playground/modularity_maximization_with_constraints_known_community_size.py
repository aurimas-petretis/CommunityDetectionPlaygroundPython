import pulp
import networkx as nx

def graph_clustering_modularity_lp(G: nx.Graph, c: int):
    nodes = list(G.nodes())
    m = G.number_of_edges()
    degrees = dict(G.degree())
    A = nx.to_dict_of_dicts(G)  # Adjacency lookup

    prob = pulp.LpProblem("ModularityMaximization", pulp.LpMaximize)

    # x[i][k] = 1 if node i is in cluster k
    x = pulp.LpVariable.dicts("x", ((i, k) for i in nodes for k in range(c)), cat='Binary')

    # y[i][j] = 1 if nodes i and j are in the same cluster
    y = pulp.LpVariable.dicts("y", ((i, j) for i in nodes for j in nodes if i < j), cat='Binary')

    # Objective: maximize modularity
    modularity_terms = []
    for i in nodes:
        for j in nodes:
            if i < j:
                Aij = 1 if j in A[i] else 0
                ki, kj = degrees[i], degrees[j]
                term = (Aij - (ki * kj) / (2 * m)) * y[(i, j)]
                modularity_terms.append(term)

    prob += (1 / (2 * m)) * pulp.lpSum(modularity_terms)

    # Constraint: each node in exactly one cluster
    for i in nodes:
        prob += pulp.lpSum(x[i, k] for k in range(c)) == 1, f"AssignNode_{i}"

    # Constraint: each cluster has at least one node
    for k in range(c):
        prob += pulp.lpSum(x[i, k] for i in nodes) >= 1, f"NonEmptyCluster_{k}"

    # Constraints to define y[i,j] based on x[i,k]
    for i in nodes:
        for j in nodes:
            if i < j:
                for k in range(c):
                    prob += y[i, j] >= x[i, k] + x[j, k] - 1, f"y_lb_{i}_{j}_{k}"
                    prob += y[i, j] <= x[i, k], f"y_ub1_{i}_{j}_{k}"
                    prob += y[i, j] <= x[j, k], f"y_ub2_{i}_{j}_{k}"

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    print("\nAssignments:")
    for i in nodes:
        for k in range(c):
            print(f"x[{i}][{k}] = {pulp.value(x[i, k])}")

    status = pulp.LpStatus[prob.status]
    print("Solver Status:", status)

    # Extract solution
    clusters = {k: [] for k in range(c)}
    for i in nodes:
        for k in range(c):
            if pulp.value(x[i, k]) is not None and pulp.value(x[i, k]) > 0.5:
                clusters[k].append(i)
                break

    return clusters




import networkx as nx

G = nx.karate_club_graph()
c = 2

clusters = graph_clustering_modularity_lp(G, c)
for k, members in clusters.items():
    print(f"Cluster {k}: {members}")