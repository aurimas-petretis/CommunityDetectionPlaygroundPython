import igraph as ig
import matplotlib.pyplot as plt

# Load the Zachary Karate Club graph
graph = ig.Graph.Famous("Zachary")

# Find the optimal modularity clustering
optimal_clustering = graph.community_optimal_modularity()

# Print the membership of each node
print("Cluster membership:")
for node, cluster in enumerate(optimal_clustering.membership):
    print(f"Node {node}: Cluster {cluster}")

# Visualize the clustering
color_palette = ig.drawing.colors.ClusterColoringPalette(len(set(optimal_clustering.membership)))
node_colors = [color_palette.get(cluster) for cluster in optimal_clustering.membership]

fig, ax = plt.subplots(figsize=(8, 6))
ig.plot(graph, target=ax, vertex_color=node_colors, vertex_size=20, edge_width=0.5)
plt.show()
