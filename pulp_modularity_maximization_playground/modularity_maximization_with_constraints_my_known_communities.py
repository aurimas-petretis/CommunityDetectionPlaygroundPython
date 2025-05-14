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


def define_variable_on_each_vertex_pair(G, c):
    x = {}
    for i in G.nodes:
        for ci in range(c):
            x[i, ci] = pulp.LpVariable(f"x_{i}_{ci}", 0, 1, pulp.LpBinary)
    return x


def define_objective(prob, G, m, degrees, x, c):
    Q = pulp.lpSum(
        (G.has_edge(i, j) - degrees[i] * degrees[j] / (2 * m)) * vertex_pair_belongs_to_the_same_cluster(x, i, j, c)
        for i in G.nodes for j in G.nodes if i < j
    )/(2*m)
    prob += Q
    return prob

def vertex_pair_belongs_to_the_same_cluster(x, i, j, c):
    for ci in range(c):
        if x[i, ci] == x[j, ci]:
            return 1
    return 0


def define_constraints(prob, x):
    # TODO add constraints
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
    print('Solution', x.items())
    # Build communities using vertex to cluster assignments from x[i,ci]
    pair_graph = nx.Graph()
    pair_graph.add_nodes_from(G.nodes)
    for (i, j), var in x.items():
        print('pulpvaluevar', i, j, pulp.value(var))
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
    c = 4
    x = define_variable_on_each_vertex_pair(G, c)
    prob = create_lp_problem()
    prob = define_objective(prob, G, m, degrees, x, c)
    prob = define_constraints(prob, x)

    prob.solve()

    communities = extract_communities(x, G)
    print_status(prob, x, communities)
    draw_solution(G, communities)


main()