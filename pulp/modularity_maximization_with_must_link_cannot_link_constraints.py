import random

import numpy as np
import partd.file
import pulp
import networkx as nx
import matplotlib.pyplot as plt


def get_zachary_graph():
    G = nx.karate_club_graph()
    n = len(G.nodes)
    m = G.number_of_edges()
    degrees = dict(G.degree)
    return G, n, m, degrees

def get_zachary_example_must_link_cannot_link_constraints():
    # must_link = [(0, 5), (0, 31)]
    # cannot_link = [(31, 33)]
    # must_link = [(0, 31)]
    # cannot_link = []
    # must_link = []
    # cannot_link = [(24, 31)]
    # return must_link, cannot_link
    return [], []


def create_lp_problem():
   return pulp.LpProblem("ModularityMaximization", pulp.LpMaximize)


def define_variable_on_each_vertex_pair(G):
    x = {}
    for i in G.nodes:
        for j in G.nodes:
            if i < j:
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", 0, 1, pulp.LpBinary)
    return x


def define_objective(prob, G, m, degrees, x):
    Q = pulp.lpSum(
        (G.has_edge(i, j) - degrees[i] * degrees[j] / (2 * m)) * x[i, j]
        for i in G.nodes for j in G.nodes if i < j
    )/(2*m)
    prob += Q
    return prob


def define_constraints(prob, G, x, n, must_link, cannot_link):
    # # constraint 0: ensured transitivity
    # for k in range(n):
    #     for j in range(k):
    #         for i in range(j):
    #             prob += x[i, j] + x[j, k] - x[i, k] >= 0
    #             prob += x[i, j] - x[j, k] + x[i, k] >= 0
    #             prob += -x[i, j] + x[j, k] + x[i, k] >= 0

    # constraint 1: there should be no more than b1 detected vertex pairs
    # b1 = n - 10
    b1 = n - 2
    prob += pulp.lpSum(x[i, j] for (i, j) in x) <= b1

    # constraint 2: vertex pairs must end up in the same cluster
    for (i, j) in must_link:
        prob += x[i, j] == 1

    # constraint 3: vertex pairs cannot end up in the same cluster
    for (i, j) in cannot_link:
        prob += x[i, j] == 0

    return prob


def print_status(prob, x, G, communities):
    print("Status:", pulp.LpStatus[prob.status])
    print("Modularity value:", pulp.value(prob.objective))
    pairs = {}
    for (i, j), var in x.items():
        if pulp.value(var) > 0:
            pairs[(i, j)] = pulp.value(var)

    community_size_average = 0
    community_size_variance = 0
    community_size_score = 0
    community_sizes = []
    for com in communities:
        community_size_average += len(com)
        community_sizes.append(len(com))
    community_size_average /= len(communities)
    print("Average community size:", community_size_average)
    for size in community_sizes:
        community_size_variance += (size - community_size_average) ** 2
        community_size_score += size ** 2
    community_size_variance /= (len(community_sizes) - 1)
    community_size_score /= (len(G.nodes) ** 2)
    print("Community size variance:", community_size_variance)
    print("Community size score:", community_size_score)

    print("Found vertex pairs:", pairs)
    print("Communities:", communities)
    print("Number of communities: ", len(communities))

def append_results_to_csv(filename, i, prob, x, G, communities, must_link, cannot_link):
    import csv

    community_sizes = []
    community_size_average = 0
    community_size_variance = 0
    community_size_score = 0
    for com in communities:
        community_size_average += len(com)
        community_sizes.append(len(com))
    community_size_average /= len(communities)
    print("Average community size:", community_size_average)
    for size in community_sizes:
        community_size_variance += (size - community_size_average) ** 2
        community_size_score += size ** 2
    community_size_variance /= (len(community_sizes) - 1)
    community_size_score /= (len(G.nodes) ** 2)
    print("Community size variance:", community_size_variance)
    print("Community size score:", community_size_score)

    csv_row = [len(must_link), i, pulp.value(prob.objective), len(communities), community_size_average, community_size_variance, community_size_score]
    with open(filename, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(csv_row)

def create_csv(filename):
    import csv
    csv_row = 'Must-link edges, Index, Modularity, Number of communities, Average Community Size, Average Community Variance, Average Community Score\n\n'
    with open(filename, 'w') as fd:
        fd.write(csv_row)


def extract_communities(x, G):
    from collections import defaultdict

    # Build communities using connected components from x[i,j]
    pair_graph = nx.Graph()
    pair_graph.add_nodes_from(G.nodes)
    for (i, j), var in x.items():
        if pulp.value(var) > 0.5:
            pair_graph.add_edge(i, j)

    # Get communities
    communities = list(nx.connected_components(pair_graph))
    return communities


def draw_solution(G, communities, must_link, cannot_link):
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map[node] for node in G.nodes]

    plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency

    # Draw the full graph normally
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.Set3, node_size=300)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray')

    # Draw must-link edges in blue
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(i, j) for (i, j) in must_link if G.has_edge(i, j)],
        edge_color='blue', width=2, style='solid'
    )

    # Draw cannot-link edges in red
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(i, j) for (i, j) in cannot_link if G.has_edge(i, j)],
        edge_color='red', width=2, style='dashed'
    )

    plt.title("Modularity Maximization with Must-Link Constraints (blue)")
    plt.axis('off')
    plt.savefig("example_modularity_maximization.png")
    plt.show()


def draw_big_solution(G, communities, must_link, cannot_link):
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map[node] for node in G.nodes]

    plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency

    # Draw the full graph normally
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.Set3, node_size=10)
    # nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray')

    # Draw must-link edges in blue
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(i, j) for (i, j) in must_link if G.has_edge(i, j)],
        edge_color='blue', width=2, style='solid'
    )

    # Draw cannot-link edges in red
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(i, j) for (i, j) in cannot_link if G.has_edge(i, j)],
        edge_color='red', width=2, style='dashed'
    )

    plt.title("Modularity Maximization with Must-Link Constraints (blue)")
    plt.axis('off')
    # plt.savefig("lfr_benchmark_modularity_maximization.png")
    plt.show()


def run_zachary_example():
    G, n, m, degrees = get_zachary_graph()
    must_link, cannot_link = get_zachary_example_must_link_cannot_link_constraints()
    x = define_variable_on_each_vertex_pair(G)
    prob = create_lp_problem()
    prob = define_objective(prob, G, m, degrees, x)
    prob = define_constraints(prob, G, x, n, must_link, cannot_link)

    prob.solve()

    communities = extract_communities(x, G)
    print_status(prob, x, communities)
    draw_solution(G, communities, must_link, cannot_link)


def run_uniform_experiment():
    n = 10
    p = 0.5
    for mlc in range(0, 2):
        for i in range(1):
            G = nx.gnp_random_graph(n, p)
            n = len(G.nodes)
            m = G.number_of_edges()
            degrees = dict(G.degree)

            must_link = []
            cannot_link = [] # unused for now
            for mli in range(mlc):
                must_link.append(random.choice([e for e in G.edges]))
            x = define_variable_on_each_vertex_pair(G)
            prob = create_lp_problem()
            prob = define_objective(prob, G, m, degrees, x)
            prob = define_constraints(prob, G, x, n, must_link, cannot_link)

            prob.solve()

            communities = extract_communities(x, G)
            print_status(prob, x, communities)
            draw_big_solution(G, communities, must_link, cannot_link)


def run_powerlaw_experiment():
    n = 15
    mg = 3
    p = 0.8
    for mlc in range(0, 2):
        for i in range(1):
            G = nx.powerlaw_cluster_graph(n, mg, p)
            n = len(G.nodes)
            m = G.number_of_edges()
            degrees = dict(G.degree)

            must_link = []
            cannot_link = [] # unused for now
            for mli in range(mlc):
                must_link.append(random.choice([e for e in G.edges]))
            x = define_variable_on_each_vertex_pair(G)
            prob = create_lp_problem()
            prob = define_objective(prob, G, m, degrees, x)
            prob = define_constraints(prob, G, x, n, must_link, cannot_link)

            prob.solve()

            communities = extract_communities(x, G)
            print_status(prob, x, communities)
            draw_big_solution(G, communities, must_link, cannot_link)

def run_LFR_benchmark_experiment():
    # https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.LFR_benchmark_graph.html
    # https://arxiv.org/pdf/0805.4770
    filename = 'lfr_benchmark_with_must_link_results.csv'
    create_csv(filename=filename)
    for mlc in range(0, 1):
        for i in range(10):
            # G = nx.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree, seed=42)
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

            n = len(G.nodes)
            m = G.number_of_edges()
            degrees = dict(G.degree)

            must_link = []
            cannot_link = [] # unused for now
            for mli in range(mlc):
                must_link.append(random.choice([e for e in G.edges]))
            x = define_variable_on_each_vertex_pair(G)
            prob = create_lp_problem()
            prob = define_objective(prob, G, m, degrees, x)
            prob = define_constraints(prob, G, x, n, must_link, cannot_link)

            prob.solve()

            communities = extract_communities(x, G)
            print_status(prob, x, G, communities)
            append_results_to_csv(filename, i, prob, x, G, communities, must_link, cannot_link)
            # draw_big_solution(G, communities, must_link, cannot_link)


def main():
    print('main')
    # run_zachary_example()
    # run_uniform_experiment()
    # run_powerlaw_experiment()
    run_LFR_benchmark_experiment()
    # one more option: https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.gaussian_random_partition_graph.html


main()