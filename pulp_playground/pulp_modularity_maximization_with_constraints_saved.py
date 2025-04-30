import igraph as ig
from pulp import *
import matplotlib.pyplot as plt

# Load graph
graph = ig.Graph.Famous("Zachary")
n = graph.vcount()
m = graph.ecount()
degrees = graph.degree()
A = graph.get_adjacency()
edges = graph.get_edgelist()
nodes = range(n)
n_clusters = 5
cluster_ids = range(n_clusters)

prob = LpProblem("myModularityMaximization", LpMaximize)
xnc = LpVariable.dicts("x", (nodes, cluster_ids), cat='Binary')
# used_clusters = LpVariable.dicts("used", cluster_ids, cat='Binary')

# TODO fix
def kronecker(i, j):
    for c in cluster_ids:
        if xnc[i][c] == xnc[j][c] == 1:
            return 1
    return 0

B = {(i, j): A[i][j] - degrees[i]*degrees[j]/(2*m) for i in range(n) for j in range(i+1, n)}
prob += lpSum(B[i, j] * kronecker(i, j) for i in range(n) for j in range(i+1, n))

# set the first node to cluster 0 to constrain starting point
prob += xnc[0][0] == 1

# Every node uses one color
for n in nodes:
    prob += lpSum([xnc[n][c] for c in cluster_ids]) == 1

prob.solve()

# Extract solution
node_to_cluster = {}
for n in nodes:
    for c in cluster_ids:
        if xnc[n][c].varValue == 1:
            node_to_cluster[n] = c

print(node_to_cluster)

# Visualize the clustering
color_palette = ig.drawing.colors.ClusterColoringPalette(len(set(node_to_cluster.values())))
node_colors = [color_palette.get(cluster_color) for cluster_color in node_to_cluster.values()]

fig, ax = plt.subplots(figsize=(8, 6))
ig.plot(graph, target=ax, vertex_color=node_colors, vertex_size=20, edge_width=0.5)
plt.show()
