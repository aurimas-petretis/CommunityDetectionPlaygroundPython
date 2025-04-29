# optimization process https://coin-or.github.io/pulp/main/the_optimisation_process.html
# elastic constraints https://coin-or.github.io/pulp/guides/how_to_elastic_constraints.html

import igraph as ig
from pulp import *

# Load graph
graph = ig.Graph.Famous("Zachary")
n = graph.vcount()
m = graph.ecount()
degrees = graph.degree()
A = graph.get_adjacency()

# LP Problem
prob = LpProblem("myModularityMaximization", LpMaximize)

def happiness(cluster):
    return lpSum(abs(i - j) * cluster[i, j] for (i, j) in cluster)

vertexes = [i for i in range(5)]
max_cluster_size = 3
possible_clusters = [tuple(c) for c in allcombinations(vertexes, 10)]
x = pulp.LpVariable.dicts(
    "cluster", possible_clusters, lowBound=0, upBound=1, cat=LpInteger
)
prob += lpSum([happiness(cluster) * x[cluster] for cluster in possible_clusters])

# Variables
# x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]
# z = {}
# for i in range(n):
#     for j in range(i+1, n):
#         z[i, j] = LpVariable(f"z_{i}_{j}", cat=LpBinary)


# Objective function
# B = {(i, j): A[i][j] - degrees[i]*degrees[j]/(2*m) for i in range(n) for j in range(i+1, n)}
# prob += lpSum(abs(i - j) * z[i, j] for (i, j) in z)
# prob += lpSum(B[i, j] * z[i, j] for (i, j) in z)

# Constraints to enforce z_{ij} = 1 if x_i == x_j
# for (i, j) in z:
#     prob += z[i, j] <= x[i] + x[j]
#     prob += z[i, j] <= 2 - x[i] - x[j]
#     prob += z[i, j] >= x[i] + x[j] - 1
#     prob += z[i, j] >= 1 - x[i] - x[j]


# Solve
status = prob.solve()

print(LpStatus[status])
for element in x:
    print(value(element))

# Get results
# membership = [int(x[i].varValue) for i in range(n)]
# print("Custom LP clustering membership:")
# for node, cluster in enumerate(membership):
#     print(f"Node {node}: Cluster {cluster}")