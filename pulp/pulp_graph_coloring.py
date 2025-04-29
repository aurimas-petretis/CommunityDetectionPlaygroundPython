# example with coloring problem based on https://stackoverflow.com/questions/65030160/how-to-speed-up-graph-coloring-problem-in-python-pulp

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
n_colors = 5
color_ids = range(n_colors)

prob = LpProblem("coloring", LpMinimize)
# variable xnc shows if node n has color c
xnc = LpVariable.dicts("x", (nodes, color_ids), cat='Binary')
# array of colors to indicate which ones were used
used_colors = LpVariable.dicts("used", color_ids, cat='Binary')

# minimize how many colors are used, and minimize int value for those colors
# prob += lpSum([used_colors[c] * c for c in colors])
prob += lpSum([used_colors[c] for c in color_ids])

# set the first node to color 0 to constrain starting point
prob += xnc[0][0] == 1

# Every node uses one color
for n in nodes:
    prob += lpSum([xnc[n][c] for c in color_ids]) == 1

# Any connected nodes have different colors
for e in edges:
    e1, e2 = e[0], e[1]
    for c in color_ids:
        prob += xnc[e1][c] + xnc[e2][c] <= 1

# mark color as used if node has that color
for n in nodes:
    for c in color_ids:
        prob += xnc[n][c] <= used_colors[c]


prob.solve()


# Extract solution
node_to_color = {}
for n in nodes:
    for c in color_ids:
        if xnc[n][c].varValue == 1:
            node_to_color[n] = c

print(node_to_color)

# Visualize the coloring
color_palette = ig.drawing.colors.ClusterColoringPalette(len(set(node_to_color.values())))
node_colors = [color_palette.get(color_color) for color_color in node_to_color.values()]

fig, ax = plt.subplots(figsize=(8, 6))
ig.plot(graph, target=ax, vertex_color=node_colors, vertex_size=20, edge_width=0.5)
plt.show()
