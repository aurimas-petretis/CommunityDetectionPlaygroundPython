import pulp
import networkx as nx
import matplotlib.pyplot as plt

# Sample graph
G = nx.karate_club_graph()
n = len(G.nodes)
m = G.number_of_edges()
degrees = dict(G.degree)

# Create the LP problem
prob = pulp.LpProblem("Modularity_Maximization", pulp.LpMaximize)

# Binary variables: x_ij = 1 if i and j are in the same community
x = {}
for i in G.nodes:
    for j in G.nodes:
        if i < j:
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", 0, 1, pulp.LpBinary)

# Force at least two communities by preventing a single complete group
prob += pulp.lpSum(x[i, j] for (i, j) in x) <= (n * (n - 1)) // 2 - 1

# Objective function: sum over all pairs (i < j)
Q = pulp.lpSum(
    (G.has_edge(i, j) - degrees[i] * degrees[j] / (2 * m)) * x[i, j]
    for i in G.nodes for j in G.nodes if i < j
)
prob += Q

# Optional: transitivity constraints (if desired for consistency)

# Solve
prob.solve()
print("Status:", pulp.LpStatus[prob.status])
print("Modularity value:", pulp.value(prob.objective))



from collections import defaultdict

# Build communities using connected components from x[i,j]
import networkx as nx

pair_graph = nx.Graph()
pair_graph.add_nodes_from(G.nodes)
for (i, j), var in x.items():
    if pulp.value(var) > 0.5:
        pair_graph.add_edge(i, j)

# Get communities
communities = list(nx.connected_components(pair_graph))

# Map node to community index
node_color_map = {}
for idx, community in enumerate(communities):
    for node in community:
        node_color_map[node] = idx

colors = [node_color_map[node] for node in G.nodes]

# Draw
plt.figure(figsize=(8, 6))
nx.draw(G, node_color=colors, with_labels=True, cmap=plt.cm.Set3)
plt.title("Modularity Maximization Result (Karate Club Graph)")
plt.show()