import igraph as ig
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary

# Load graph
graph = ig.Graph.Famous("Zachary")
n = graph.vcount()
m = graph.ecount()
degrees = graph.degree()
A = graph.get_adjacency()

# LP Problem
prob = LpProblem("ModularityMaximization", LpMaximize)

# Variables
x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]
z = {}
for i in range(n):
    for j in range(i+1, n):
        z[i, j] = LpVariable(f"z_{i}_{j}", cat=LpBinary)


# Objective function
B = {(i, j): A[i][j] - degrees[i]*degrees[j]/(2*m) for i in range(n) for j in range(i+1, n)}
prob += lpSum(B[i, j] * z[i, j] for (i, j) in z)

# Constraints to enforce z_{ij} = 1 if x_i == x_j
for (i, j) in z:
    prob += z[i, j] <= x[i] + x[j]
    prob += z[i, j] <= 2 - x[i] - x[j]
    prob += z[i, j] >= x[i] + x[j] - 1
    prob += z[i, j] >= 1 - x[i] - x[j]


# Solve
prob.solve()

# Get results
membership = [int(x[i].varValue) for i in range(n)]
print("Custom LP clustering membership:")
for node, cluster in enumerate(membership):
    print(f"Node {node}: Cluster {cluster}")