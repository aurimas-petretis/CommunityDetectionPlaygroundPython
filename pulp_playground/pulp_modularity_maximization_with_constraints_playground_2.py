# Re-import required libraries after code execution state reset
import networkx as nx
import matplotlib.pyplot as plt
import pulp

# Create the Karate Club graph
G = nx.karate_club_graph()
n = len(G.nodes)
m = G.number_of_edges()
degrees = dict(G.degree)

# Create the LP problem
prob = pulp.LpProblem("Modularity_Maximization_NodeBased", pulp.LpMaximize)

# Integer variables: c_i = community assignment for node i
c = {i: pulp.LpVariable(f"c_{i}", 0, n - 1, cat="Integer") for i in G.nodes}

# Binary variables: same_comm[i,j] = 1 if nodes i and j are in the same community
same_comm = {}
for i in G.nodes:
    for j in G.nodes:
        if i < j:
            same_comm[i, j] = pulp.LpVariable(f"same_{i}_{j}", 0, 1, cat="Binary")
            # Big-M constraints to enforce same_comm = 1 <=> c[i] == c[j]
            prob += c[i] - c[j] <= (1 - same_comm[i, j]) * n
            prob += c[j] - c[i] <= (1 - same_comm[i, j]) * n

# Objective function: maximize modularity
Q = pulp.lpSum(
    (G.has_edge(i, j) - degrees[i] * degrees[j] / (2 * m)) * same_comm[i, j]
    for i in G.nodes for j in G.nodes if i < j
)
prob += Q

# Constraint: ensure at least two communities (not all nodes same)
prob += pulp.lpSum(same_comm[i, j] for i in G.nodes for j in G.nodes if i < j) <= (n * (n - 1)) // 2 - 1
# prob += pulp_playground.lpSum(same_comm[i, j] for i in G.nodes for j in G.nodes if i < j) <= 5

# Solve the problem
prob.solve()

# Extract community assignments
community_assignments = {i: int(pulp.value(var)) for i, var in c.items()}
colors = [community_assignments[i] for i in G.nodes]

# Visualize the graph with nodes colored by community
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_color=colors, with_labels=True, cmap=plt.cm.Set3, node_size=500, edge_color='gray')
plt.title("Modularity Maximization with Explicit Community Assignments")
plt.show()
