
import networkx as nx

def no_surgery(G_origin, *args, **kwargs):
    return G_origin

def surgery(G_origin, weight='weight', cut_proportion=0.1):
    G = G_origin.copy()
    w = nx.get_edge_attributes(G, weight)

    assert cut_proportion >= 0 and cut_proportion <= 1, "Cut proportion should be in [0, 1]"

    sorted_edges = sorted(w.items(), key=lambda x:x[1])
    to_cut = [e for (e, e_w) in sorted_edges[int(len(sorted_edges) * (1 - cut_proportion)):]]
    G.remove_edges_from(to_cut)
    return G

def surgery_n(G_origin, weight='weight', cut_n=1):
    G = G_origin.copy()
    w = nx.get_edge_attributes(G, weight)

    assert cut_n >= 0 and cut_n <= G.number_of_edges(), "Cut proportion should be in [0, 1]"

    sorted_edges = sorted(w.items(), key=lambda x:x[1])
    # to_cut = []
    # for e, e_w in sorted_edges[-cut_n:]:
    #     if e_w < 0:
    #         to_cut.append(e)
    cut_n = int(cut_n)
    to_cut = [e for (e, e_w) in sorted_edges[-cut_n:]]
    G.remove_edges_from(to_cut)
    return G

def my_surgery(G_origin: nx.Graph(), weight="weight", cut=10):
    
    G = G_origin.copy()
    w = nx.get_edge_attributes(G, weight)

    assert cut >= 0, "Cut value should be greater than 0."
    if not cut:
        cut = (max(w.values()) - 1.0) * 0.6 + 1.0  # Guess a cut point as default

    to_cut = []
    for n1, n2 in G.edges():
        if G[n1][n2][weight] > cut:
            to_cut.append((n1, n2))
    print("*************** Surgery time ****************")
    print("* Cut %d edges." % len(to_cut))
    G.remove_edges_from(to_cut)
    print("* Number of nodes now: %d" % G.number_of_nodes())
    print("* Number of edges now: %d" % G.number_of_edges())
    cc = list(nx.connected_components(G))
    print("* Modularity now: %f " % nx.algorithms.community.quality.modularity(G, cc))
    print("* ARI now: %f " % ARI(G, cc))
    print("*********************************************")

    return G