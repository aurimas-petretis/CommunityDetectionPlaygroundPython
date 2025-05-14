import pulp
import networkx as nx
import matplotlib.pyplot as plt


def get_zachary_graph():
    G = nx.karate_club_graph()
    n = len(G.nodes)
    m = G.number_of_edges()
    degrees = dict(G.degree)
    return G, n, m, degrees


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
    b1 = n
    prob += pulp.lpSum(x[i, j] for (i, j) in x) <= b1

    # constraint 2: vertex pairs must end up in the same cluster
    for (i, j) in must_link:
        prob += x[i, j] == 1

    # constraint 3: vertex pairs cannot end up in the same cluster
    for (i, j) in cannot_link:
        prob += x[i, j] == 0

    return prob


def print_status(prob, x, communities):
    print("Status:", pulp.LpStatus[prob.status])
    print("Modularity value:", pulp.value(prob.objective))
    pairs = {}
    for (i, j), var in x.items():
        if pulp.value(var) > 0:
            pairs[(i, j)] = pulp.value(var)
    print("Found vertex pairs:", pairs)
    print("Communities:", communities)


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

    plt.title("Modularity Maximization with Must-Link (blue) and Cannot-Link (red)")
    plt.axis('off')
    plt.savefig("zachary_modularity_maximization_one_cannot_link_constraint.png")
    plt.show()


def main():
    G, n, m, degrees = get_zachary_graph()
    # must_link = [(0, 5), (0, 31)]
    # cannot_link = [(31, 33)]
    must_link, cannot_link = [], []
    # must_link = [(0, 31)]
    # cannot_link = []
    must_link = []
    cannot_link = [(24, 31)]
    print(n, m, must_link, cannot_link, degrees)
    x = define_variable_on_each_vertex_pair(G)
    prob = create_lp_problem()
    prob = define_objective(prob, G, m, degrees, x)
    prob = define_constraints(prob, G, x, n, must_link, cannot_link)

    prob.solve()

    communities = extract_communities(x, G)
    print_status(prob, x, communities)
    draw_solution(G, communities, must_link, cannot_link)


main()