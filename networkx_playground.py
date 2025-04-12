import networkx as nx
import matplotlib.pyplot as plt


def southern_women_graph_example():
    G = nx.davis_southern_women_graph()  # Example graph
    communities = nx.community.greedy_modularity_communities(G)

    # Compute positions for the node clusters as if they were themselves nodes in a
    # supergraph using a larger scale factor
    supergraph = nx.cycle_graph(len(communities))
    superpos = nx.spring_layout(G, scale=50, seed=429)

    # Use the "supernode" positions as the center of each node cluster
    centers = list(superpos.values())
    pos = {}
    for center, comm in zip(centers, communities):
        pos.update(nx.spring_layout(nx.subgraph(G, comm), center=center, seed=1430))

    # Nodes colored by cluster
    for nodes, clr in zip(communities, ("tab:blue", "tab:orange", "tab:green")):
        nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes, node_color=clr, node_size=100)
    nx.draw_networkx_edges(G, pos=pos)

    plt.tight_layout()
    plt.show()


def zachary_karate_clum_example():
    graph = nx.karate_club_graph()
    # visualize_graph(graph)
    return graph


def find_greedy_modularity_communities(G):
    communities = nx.community.greedy_modularity_communities(G)
    print_communities(G, communities)


def find_label_propagation_communities(G):
    communities = nx.community.label_propagation_communities(G)
    print_communities(G, communities)


def find_edge_betweenness_partition(G):
    communities = nx.community.edge_betweenness_partition(G, 2)
    print_communities(G, communities)


def print_communities(G, communities):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    for nodes, clr in zip(communities, ("tab:blue", "tab:orange", "tab:green")):
        nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes, node_color=clr)

    plt.show()


def create_example_graph_0():
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(4)])
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 4)])
    visualize_graph(graph)
    return graph


def create_example_graph_1():
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(10)])
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7), (4, 8), (4, 9), (5, 6), (6, 7), (7, 8), (8, 9)])
    visualize_graph(graph)
    return graph

def create_example_graph_2():
    graph = create_example_graph_1()
    graph.add_edge(3, 4)
    visualize_graph(graph)
    return graph


def visualize_graph(graph):
    pos = nx.spring_layout(graph)  # Position nodes with a spring layout
    nx.draw(graph, pos, with_labels=True, font_weight='bold')
    # Extract edge weights and display them as labels
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()


def main():
    print('Starting program')

    # southern_women_graph_example()
    karate_club_graph = zachary_karate_clum_example()
    # find_greedy_modularity_communities(karate_club_graph)
    # find_label_propagation_communities(karate_club_graph)
    find_edge_betweenness_partition(karate_club_graph)

    # graph_0 = create_example_graph_0()
    # graph_1 = create_example_graph_1()
    # graph_2 = create_example_graph_2()
    # find_greedy_modularity_communities(graph_0)
    # find_greedy_modularity_communities(graph_1)
    # find_greedy_modularity_communities(graph_2)


    print('Ending program')

main()