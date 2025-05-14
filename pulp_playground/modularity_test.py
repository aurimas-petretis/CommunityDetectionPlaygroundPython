import pulp
import networkx as nx
import matplotlib.pyplot as plt

def node_to_cluster_modularity(G, C, must_link=None, cannot_link=None):
    """
    Node-to-cluster formulation for modularity maximization.

    Args:
        G           : NetworkX graph
        C           : maximum number of clusters
        must_link   : list of (i,j) pairs that must share a cluster
        cannot_link : list of (i,j) pairs that cannot share a cluster
    Returns:
        List of communities (each a list of nodes), length exactly C.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    nodes = list(G.nodes())
    degrees = dict(G.degree())

    # create problem
    prob = pulp.LpProblem("Modularity_Max_NodeCluster", pulp.LpMaximize)

    # y[i,c] = 1 if node i assigned to cluster c
    y = pulp.LpVariable.dicts("y", (nodes, range(C)), 0, 1, pulp.LpBinary)

    # each node belongs to exactly one cluster
    for i in nodes:
        prob += pulp.lpSum(y[i][c] for c in range(C)) == 1, f"one_cluster_{i}"

    # z[i,j] = 1 if nodes i and j share a cluster (linearized)
    z = {}
    for i in nodes:
        for j in nodes:
            if i < j:
                z[(i, j)] = pulp.LpVariable(f"z_{i}_{j}", 0, 1, pulp.LpBinary)
                for c in range(C):
                    prob += z[(i, j)] <= y[i][c]
                    prob += z[(i, j)] <= y[j][c]
                    prob += z[(i, j)] >= y[i][c] + y[j][c] - 1

    # modularity objective
    prob += pulp.lpSum(
        ((1 if G.has_edge(i, j) else 0) - degrees[i] * degrees[j] / (2*m)) * z[(i, j)]
        for (i, j) in z
    ) / (2*m)

    # must-link: force z[i,j] == 1
    if must_link:
        for (i, j) in must_link:
            a, b = (i, j) if i < j else (j, i)
            prob += z[(a, b)] == 1

    # cannot-link: force z[i,j] == 0
    if cannot_link:
        for (i, j) in cannot_link:
            a, b = (i, j) if i < j else (j, i)
            prob += z[(a, b)] == 0

    # solve
    prob.solve()

    # extract communities
    communities = [[] for _ in range(C)]
    for i in nodes:
        for c in range(C):
            if pulp.value(y[i][c]) > 0.5:
                communities[c].append(i)

    # return all clusters (some may be empty)
    return communities


def draw_communities(G, communities):
    # color nodes by community
    color_map = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            color_map[node] = cid
    colors = [color_map[n] for n in G.nodes()]

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_color=colors, with_labels=True, cmap=plt.cm.Set3)
    plt.title(f"Detected {len(communities)} communities")
    plt.axis('off')
    plt.show()


def main():
    G = nx.karate_club_graph()
    must_link = []
    cannot_link = [(24, 31)]

    # specify desired number of clusters
    C = 4
    communities = node_to_cluster_modularity(G, C, must_link=must_link, cannot_link=cannot_link)

    print(f"Detected communities ({len(communities)}): {communities}")
    draw_communities(G, communities)

if __name__ == '__main__':
    main()
