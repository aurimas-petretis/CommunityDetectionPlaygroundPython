import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_graph(G, title, node_community_map):
    plt.figure(figsize=(7, 7))
    pos = nx.spring_layout(G, seed=42)

    # Assign a unique color to each community
    communities = defaultdict(list)
    for node, comm in node_community_map.items():
        communities[comm].append(node)

    colors = plt.get_cmap('tab10').colors
    for i, nodes in enumerate(communities.values()):
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=60,
                               node_color=[colors[i % len(colors)]], label=f"Community {i+1}")
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.legend()
    plt.show()

# Generate non-overlapping LFR graph
G = nx.LFR_benchmark_graph(
    n=250,
    tau1=3,
    tau2=1.5,
    mu=0.1,
    average_degree=5,
    max_degree=50,
    min_community=20,
    max_community=100,
    seed=42
)

# Ensure undirected and clean
G = G.to_undirected()
G.remove_edges_from(nx.selfloop_edges(G))

# Extract ground-truth communities (non-overlapping: each node has 1 community)
node_community_map = {}
for node, data in G.nodes(data=True):
    # Select the first community only (guaranteed to be one in this config)
    community = list(data['community'])[0]
    node_community_map[node] = community

# Plot subset for clarity
subset_nodes = list(G.nodes)[:80]
plot_graph(G.subgraph(subset_nodes), "LFR Benchmark Graph (Non-overlapping, Subset)", node_community_map)
