import numpy as np
import networkx as nx
import pulp
from typing import List, Tuple, Dict


def modularity_maximization_ilp(graph: nx.Graph, max_communities: int = None) -> Dict[int, int]:
    """
    Solve the modularity maximization problem using Integer Linear Programming (ILP).

    Args:
        graph: NetworkX graph
        max_communities: Maximum number of communities to consider (optional)

    Returns:
        Dictionary mapping nodes to their community assignments
    """
    # Get basic graph properties
    nodes = list(graph.nodes())
    num_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # If max_communities is not specified, use number of nodes as upper bound
    if max_communities is None:
        max_communities = num_nodes

    # Calculate total number of edges (m) and degree of each node
    m = graph.number_of_edges()
    degrees = dict(graph.degree())

    # Create the ILP problem
    prob = pulp.LpProblem("Modularity_Maximization", pulp.LpMaximize)

    # Decision variables
    # x_ik = 1 if node i is in community k, 0 otherwise
    x = pulp.LpVariable.dicts("x",
                              ((i, k) for i in range(num_nodes)
                               for k in range(max_communities)),
                              cat='Binary')

    # y_ij = 1 if nodes i and j are in the same community, 0 otherwise
    y = pulp.LpVariable.dicts("y",
                              ((i, j) for i in range(num_nodes)
                               for j in range(i + 1, num_nodes)),
                              cat='Binary')

    # z_ijk = x_ik * x_jk (product linearization)
    z = pulp.LpVariable.dicts("z",
                              ((i, j, k) for i in range(num_nodes)
                               for j in range(i + 1, num_nodes)
                               for k in range(max_communities)),
                              cat='Binary')

    # Objective function: Maximize modularity
    # Q = 1/(2m) * sum_ij [A_ij - (k_i*k_j)/(2m)] * delta(c_i, c_j)
    # We'll use y_ij to represent delta(c_i, c_j)

    # First part: sum over edges
    edge_sum = pulp.lpSum(y[i, j] * (1 - (degrees[nodes[i]] * degrees[nodes[j]]) / (2 * m))
                          for i in range(num_nodes)
                          for j in range(i + 1, num_nodes)
                          if graph.has_edge(nodes[i], nodes[j]))

    # Second part: sum over non-edges
    non_edge_sum = pulp.lpSum(y[i, j] * (- (degrees[nodes[i]] * degrees[nodes[j]]) / (2 * m))
                              for i in range(num_nodes)
                              for j in range(i + 1, num_nodes)
                              if not graph.has_edge(nodes[i], nodes[j]))

    prob += (edge_sum + non_edge_sum) / (2 * m), "Modularity"

    # Constraints

    # Each node must be in exactly one community
    for i in range(num_nodes):
        prob += pulp.lpSum(x[i, k] for k in range(max_communities)) == 1, f"Node_{i}_in_one_community"

    # Link x and y variables using z_ijk = x_ik * x_jk
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            for k in range(max_communities):
                # z_ijk <= x_ik
                prob += z[i, j, k] <= x[i, k], f"z_leq_xik_{i}_{j}_{k}"
                # z_ijk <= x_jk
                prob += z[i, j, k] <= x[j, k], f"z_leq_xjk_{i}_{j}_{k}"
                # z_ijk >= x_ik + x_jk - 1
                prob += z[i, j, k] >= x[i, k] + x[j, k] - 1, f"z_geq_xik_plus_xjk_minus_1_{i}_{j}_{k}"

            # y_ij = sum_k z_ijk
            prob += y[i, j] == pulp.lpSum(z[i, j, k] for k in range(max_communities)), f"y_eq_sum_z_{i}_{j}"

    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=True)
    prob.solve(solver)

    # Extract the solution
    communities = {}
    for i, node in enumerate(nodes):
        for k in range(max_communities):
            if pulp.value(x[i, k]) > 0.5:  # If x_ik is approximately 1
                communities[node] = k
                break

    return communities


def main():
    # Example usage
    # Create a sample graph (Zachary's karate club network)
    G = nx.karate_club_graph()

    # Add some node attributes for better visualization
    for node in G.nodes():
        G.nodes[node]['label'] = str(node)

    print("Solving modularity maximization using ILP...")
    communities = modularity_maximization_ilp(G, max_communities=5)

    # Print community assignments
    print("\nCommunity assignments:")
    for node, comm in sorted(communities.items()):
        print(f"Node {node}: Community {comm}")

    # Calculate modularity of the found partition
    try:
        import community as community_louvain
        modularity = community_louvain.modularity(communities, G)
        print(f"\nModularity of the partition: {modularity:.4f}")
    except ImportError:
        print("\nInstall python-louvain package to calculate modularity: pip install python-louvain")

    # Visualize the graph with community colors
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # Use spring layout for better visualization
        pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducible layout

        # Create a color map with distinct colors
        unique_communities = sorted(set(communities.values()))
        colors = list(mcolors.TABLEAU_COLORS.values())
        if len(unique_communities) > len(colors):
            # If we have more communities than colors, generate more colors
            cmap = plt.cm.get_cmap('tab20', len(unique_communities))
            colors = [mcolors.to_hex(cmap(i)) for i in range(len(unique_communities))]

        plt.figure(figsize=(12, 10))

        # Draw nodes with community colors
        for i, comm in enumerate(unique_communities):
            nodes_in_comm = [node for node, c in communities.items() if c == comm]
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes_in_comm,
                node_color=[colors[i % len(colors)]] * len(nodes_in_comm),
                node_size=500,
                alpha=0.9,
                label=f'Community {comm}'
            )

        # Draw edges with some transparency
        nx.draw_networkx_edges(
            G, pos,
            width=1.0,
            alpha=0.5,
            edge_color='gray'
        )

        # Draw node labels
        nx.draw_networkx_labels(
            G, pos,
            labels={n: str(n) for n in G.nodes()},
            font_size=10,
            font_weight='bold',
            font_color='white'
        )

        # Add title and legend
        plt.title("Community Detection using ILP for Modularity Maximization\nZachary's Karate Club Network",
                  fontsize=14)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), fontsize=10)

        # Add text box with statistics
        stats_text = f"Network Statistics:\n"
        stats_text += f"• Nodes: {G.number_of_nodes()}\n"
        stats_text += f"• Edges: {G.number_of_edges()}\n"
        stats_text += f"• Communities: {len(unique_communities)}"

        plt.figtext(
            0.02, 0.02,
            stats_text,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
            fontsize=10
        )

        plt.axis('off')
        plt.tight_layout()

        # Save the figure
        plt.savefig('community_detection_result.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'community_detection_result.png'")

        plt.show()

    except ImportError as e:
        print(f"\nError during visualization: {e}")
        print("Make sure you have matplotlib installed: pip install matplotlib")


if __name__ == "__main__":
    main()
