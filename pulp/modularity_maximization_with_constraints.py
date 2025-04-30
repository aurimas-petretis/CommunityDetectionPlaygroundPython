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
   return pulp.LpProblem("Modularity_Maximization", pulp.LpMaximize)


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


def define_constraints(prob, x):
    # constraint 1: there should be no more than b1 detected vertex pairs
    # TODO make b1 variable depend on graph size
    # TODO it would be better to transform this constraint to check how many communities are needed,
    #  it's difficult to determine what current formulation does represent because some some edges form cycles,
    #  might be solvable by adding another constraint
    b1 = 40
    prob += pulp.lpSum(x[i, j] for (i, j) in x) <= b1
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


def draw_solution(G, communities):
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map[node] for node in G.nodes]
    print("Vertex coloring based on communities:", colors)

    plt.figure(figsize=(8, 6))
    nx.draw(G, node_color=colors, with_labels=True, cmap=plt.cm.Set3)
    plt.title("Modularity Maximization Result")
    plt.show()


def main():
    G, n, m, degrees = get_zachary_graph()
    x = define_variable_on_each_vertex_pair(G)
    prob = create_lp_problem()
    prob = define_objective(prob, G, m, degrees, x)
    prob = define_constraints(prob, x)

    prob.solve()

    communities = extract_communities(x, G)
    print_status(prob, x, communities)
    draw_solution(G, communities)


main()