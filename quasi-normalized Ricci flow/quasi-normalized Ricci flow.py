
import numpy as np
import math
import time
from sklearn import preprocessing, metrics
from multiprocessing import Pool, cpu_count
import community.community_louvain as community_louvain
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
import importlib
import cvxpy as cvx
import urllib.request
import io
import zipfile
import surgery as Surgery
import matplotlib.pyplot as plt
import networkx as nx
from urllib.request import urlopen



# This class is used to compute the quasi-normalized Ricci curvature of a given graph.
class RhoNormalizeCurvature:
    
    def __init__(self, G, weight="weight", proc=cpu_count()):
        self.G = G
        self.weight=weight
        self.proc = proc
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}
        self.EPSILON = 1e-7  # to prevent divided by zero
        self.base = math.e
        self.exp_power = 2

    def _get_all_pairs_shortest_path(self):
        # Construct the all pair shortest path lookup
        lengths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight=self.weight))
        return lengths

    def _get_edge_density_distributions(self):
        densities = dict()

        def Gamma(i, j):
            return self.lengths[i][j]

        # Construct the density distributions on each node
        def get_single_node_neighbors_distributions(neighbors):
            # Get sum of distributions from x's all neighbors
            nbr_edge_weight_sum = sum([Gamma(x,nbr) for nbr in neighbors])

            if nbr_edge_weight_sum > self.EPSILON:
                result = [Gamma(x,nbr) / nbr_edge_weight_sum for nbr in neighbors]
            elif len(neighbors) == 0:
                return []
            else:
                result = [1.0 / len(neighbors)] * len(neighbors)
            result.append(0)
            return result

        for x in self.G.nodes():
            densities[x] = get_single_node_neighbors_distributions(list(self.G.neighbors(x)))

        return densities

    def _optimal_transportation_distance(self, x, y, d):
        star_coupling = cvx.Variable((len(y), len(x)))  # the transportation plan B
        # objective function sum(star_coupling(x,y) * d(x,y)) , need to do element-wise multiply here
        obj = cvx.Maximize(cvx.sum(cvx.multiply(star_coupling, d.T)))
        # constrains
        constrains = [cvx.sum(star_coupling)==0]

        constrains += [cvx.sum(star_coupling[:, :-1], axis=0, keepdims=True) == np.multiply(-1, x.T[:,:-1])]
        constrains += [cvx.sum(star_coupling[:-1, :], axis=1, keepdims=True) == np.multiply(-1, y[:-1])]

        constrains += [0 <= star_coupling[-1, -1], star_coupling[-1, -1] <= 2]
        constrains += [star_coupling[:-1,:-1] <= 0]
        constrains += [star_coupling[-1,:-1] <= 0]
        constrains += [star_coupling[:-1,-1] <= 0]

        prob = cvx.Problem(obj, constrains)

        m = prob.solve(solver="ECOS")  # change solver here if you want
        # solve for optimal transportation cost
        return m

    def _distribute_densities(self, source, target):

        # Append source and target node into weight distribution matrix x,y
        source_nbr = list(self.G.neighbors(source))
        target_nbr = list(self.G.neighbors(target))

        # Distribute densities for source and source's neighbors as x
        if not source_nbr:
            source_nbr.append(source)
            x = [1]
        else:
            source_nbr.append(source)
            x = self.densities[source]

        # Distribute densities for target and target's neighbors as y
        if not target_nbr:
            target_nbr.append(target)
            y = [1]
        else:
            target_nbr.append(target)
            y = self.densities[target]

        # construct the cost dictionary from x to y
        d = np.zeros((len(x), len(y)))

        for i, src in enumerate(source_nbr):
            for j, dst in enumerate(target_nbr):
                assert dst in self.lengths[src], "Target node not in list, should not happened, pair (%d, %d)" % (src, dst)
                d[i][j] = self.lengths[src][dst]

        x = np.array([x]).T  # the mass that source neighborhood initially owned
        y = np.array([y]).T  # the mass that target neighborhood needs to received

        return x, y, d

    def _compute_ricci_curvature_single_edge(self, source, target):

        assert source != target, "Self loop is not allowed."  # to prevent self loop

        # If the weight of edge is too small, return 0 instead.
        if self.lengths[source][target] < self.EPSILON:
            print("Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead." % (source, target))
            return {(source, target): 0}

        # compute transportation distance
        m = 1  # assign an initial cost

        x, y, d = self._distribute_densities(source, target)
        m = self._optimal_transportation_distance(x, y, d)

        # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
        result = m / self.lengths[source][target]  # Divided by the length of d(i, j)
        #print("Ricci curvature (%s,%s) = %f" % (source, target, result))

        return {(source, target): result}

    def _wrap_compute_single_edge(self, stuff):
        return self._compute_ricci_curvature_single_edge(*stuff)

    def compute_ricci_curvature_edges(self, edge_list=None):
        
        if not edge_list:
            edge_list = []

        # Construct the all pair shortest path dictionary
        #if not self.lengths:
        self.lengths = self._get_all_pairs_shortest_path()

        # Construct the density distribution
        if not self.densities:
            self.densities = self._get_edge_density_distributions()
        
        # Start compute edge Ricci curvature
       
            
       #     p = Pool(processes=self.proc)
            
        # Compute Ricci curvature for edges
        args = [(source, target) for source, target in edge_list]

        result = [self._wrap_compute_single_edge(arg) for arg in args]
            
        #    result = p.map_async(self._wrap_compute_single_edge, args).get()
         #   p.close()
          #  p.join()

        return result

    def compute_ricci_curvature(self):
        
        if not nx.get_edge_attributes(self.G, self.weight):
            print('Edge weight not detected in graph, use "weight" as edge weight.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = 1.0

                
        edge_ricci = self.compute_ricci_curvature_edges(self.G.edges())
        
        # Assign edge Ricci curvature from result to graph G
        for rc in edge_ricci:
            for k in list(rc.keys()):
                source, target = k
                self.G[source][target]['ricciCurvature'] = rc[k]

        # Compute node Ricci curvature
        for n in self.G.nodes():
            rc_sum = 0  # sum of the neighbor Ricci curvature
            if self.G.degree(n) != 0:
                for nbr in self.G.neighbors(n):
                    if 'ricciCurvature' in self.G[n][nbr]:
                        rc_sum += self.G[n][nbr]['ricciCurvature']

                # Assign the node Ricci curvature to be the average of node's adjacency edges
                self.G.nodes[n]['ricciCurvature'] = rc_sum / self.G.degree(n)


    def compute_ricci_flow(self, iterations=5, step=0.01, delta=1e-6, surgery={'name':'surgery', 'portion': 0.02, 'interval': 5}):
        if not nx.is_connected(self.G):
            print("Not connected graph detected, compute on the largest connected component instead.")
            self.G = nx.Graph(max([self.G.subgraph(c) for c in nx.connected_components(self.G)], key=len))
            print('---------------------------')
            print(self.G)

        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        

        
        # Start compute edge Ricci flow
        t0 = time.time()

        if nx.get_edge_attributes(self.G, "original_RC"):
            print("original_RC detected, continue to refine the ricci flow.")
        else:
            self.compute_ricci_curvature()

            for (v1, v2) in self.G.edges():
                self.G[v1][v2]["original_RC"] = self.G[v1][v2]["ricciCurvature"]

        # Start the Ricci flow process
        self.rc_diff = []
        for i in range(iterations):
            
            w = nx.get_edge_attributes(self.G, self.weight)
            
            sum_K_R = sum(self.G[v1][v2]["ricciCurvature"] * self.lengths[v1][v2] for (v1, v2) in self.G.edges())
            sumr=0
            for (v1, v2) in self.G.edges():
                sumr+=self.G[v1][v2][self.weight]
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] -= step * (self.G[v1][v2]["ricciCurvature"]) * (self.lengths[v1][v2])
                self.G[v1][v2][self.weight] += step*(sum_K_R)/sumr  
           
            #Merge really adjacent node
            G1 = self.G.copy()
            merged = True
            while merged:
                merged = False
                for v1,v2 in G1.edges():
                    if G1[v1][v2][self.weight] < delta * 10:
                        G1 = nx.contracted_edge(G1, (v1, v2), self_loops=False)
                        merged = True
                        break
            self.G = G1

            self.compute_ricci_curvature()
            print("=== Ricciflow iteration % d ===" % int(i+1))
            
            rc = nx.get_edge_attributes(self.G, "ricciCurvature")
            if rc:
                diff = max(rc.values()) - min(rc.values())
                print("Ricci curvature difference: %f" % diff)
                print("max:%f, min:%f | maxw:%f, minw:%f" % (max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))
            else:
                diff = 0

           

            if diff < delta:
                print("Ricci curvature converged, process terminated.")
                break

            # do surgery or any specific evaluation
            surgery_func = surgery['name']
            do_surgery = surgery['interval']
            portion = surgery['portion']
            if i != 0 and i % do_surgery == 0:
                self.G = getattr(Surgery, surgery_func)(self.G, self.weight, portion)
                
            # clear the APSP and densities since the graph have changed.
            self.densities = {}

      
        print("\n%8f secs for Ricci flow computation." % (time.time() - t0))

def show_results(G, curvature="ricciCurvature"):

    

    # Plot the histogram of Ricci curvatures
    plt.subplot(2, 1, 1)
    ricci_curvtures = nx.get_edge_attributes(G, curvature).values()
    plt.hist(ricci_curvtures,bins=20)
    plt.xlabel('Ricci curvature')
    plt.title("Histogram of Ricci Curvatures ")

    # Plot the histogram of edge weights
    plt.subplot(2, 1, 2)
    weights = nx.get_edge_attributes(G, "weight").values()
    plt.hist(weights,bins=20)
    plt.xlabel('Edge weight')
    plt.title("Histogram of Edge weights ")

    plt.tight_layout()
    plt.show()


    return G



def draw_graph(G, clustering_label="club"):
    """
    A helper function to draw a nx graph with community.
    """
    complex_list = nx.get_node_attributes(G, clustering_label)
    le = preprocessing.LabelEncoder()
    node_color = le.fit_transform(list(complex_list.values()))
    pos=nx.spring_layout(G)
    nx.draw_spring(G,nodelist=G.nodes(),
                   node_color=node_color,
                   cmap=plt.cm.rainbow,
                   alpha=0.8)
    plt.show()

    

def ARI(G, clustering, clustering_label="club"):

    complex_list = nx.get_node_attributes(G, clustering_label)
    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(list(complex_list.values()))

    if isinstance(clustering, dict):
        # python-louvain partition format
        y_pred = np.array([clustering[v] for v in complex_list.keys()])
    elif isinstance(clustering[0], set):
        # networkx partition format
        predict_dict = {c: idx for idx, comp in enumerate(clustering) for c in comp}
        y_pred = np.array([predict_dict[v] for v in complex_list.keys()])
    elif isinstance(clustering, list):
        # sklearn partition format
        y_pred = clustering
    else:
        return -1

    return metrics.adjusted_rand_score(y_true, y_pred)


def NMI(G, clustering, clustering_label="club"):
    
    
    complex_list = nx.get_node_attributes(G, clustering_label)
    
    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(list(complex_list.values()))
    
    if isinstance(clustering, dict):
        # python-louvain partition format
        y_pred = np.array([clustering[v] for v in complex_list.keys()])
    elif isinstance(clustering[0], set):
        # networkx partition format
        predict_dict = {c: idx for idx, comp in enumerate(clustering) for c in comp}
        y_pred = np.array([predict_dict[v] for v in complex_list.keys()])
    elif isinstance(clustering, list):
        # sklearn partition format
        y_pred = clustering
    else:
        return -1
    
    return metrics.normalized_mutual_info_score(y_true, y_pred)

#my_surgery(_rc.G, weight="weight", cut=1.0)

def check_accuracy(G_origin, weight="weight", clustering_label="club", plot_cut=True):
    """To check the clustering quality while cut the edges with weight using different threshold

    Parameters
    ----------
    G_origin : NetworkX graph
        A graph with ``weight`` as Ricci flow metric to cut.
    weight: float
        The edge weight used as Ricci flow metric. (Default value = "weight")
    clustering_label : str
        Node attribute name for ground truth.
    """
    G = G_origin.copy()
    modularity, ari ,nmi = [], [], []
    maxw = max(nx.get_edge_attributes(G, weight).values())
    cutoff_range = np.arange(maxw,0.1 , -0.01)
    for cutoff in cutoff_range:
        
        edge_trim_list = []
        for n1, n2 in G.edges():
            if G[n1][n2][weight] > cutoff:
                edge_trim_list.append((n1, n2))
        G.remove_edges_from(edge_trim_list)
        if G.number_of_edges() == 0:
            cutoff_range=np.arange(maxw,cutoff+0.01 , -0.01)
            print("No edges left in the graph. Exiting the loop.")
            break
        # Get connected component after cut as clustering
        clustering = {c: idx for idx, comp in enumerate(nx.connected_components(G)) for c in comp}
       

        # Compute modularity and ari 
        c_communities=list(nx.connected_components(G))
        modularity.append(nx.community.modularity(G, c_communities))
        
        ari.append(ARI(G, clustering, clustering_label=clustering_label))
        nmi.append(NMI(G, clustering, clustering_label=clustering_label))

    plt.xlim(maxw, 0)
    plt.xlabel("Edge weight cutoff")
    plt.plot(cutoff_range, modularity, alpha=0.8)
    plt.plot(cutoff_range, ari, alpha=0.8)
    plt.plot(cutoff_range, nmi, alpha=0.8)

    if plot_cut==False:
        plt.legend(['Modularity', 'Adjust Rand Index',"NMI"])
    
    print("MaxModularity:",max(modularity))
    print("MaxARI:",max(ari))
    print("MaxNMI:",max(nmi))
    plt.show()
    return G



    
  

def main():
   
    G = nx.read_gexf(r"C:\Users\Administrator\Desktop\Modified Ricci Flow\data\football.gexf")
    
    G_1 = G.copy()
    
    _rnc = RhoNormalizeCurvature(G_1)
    _rnc.compute_ricci_curvature()
    
    
    show_results(_rnc.G)
   
    _rnc.compute_ricci_flow(iterations=30, step=0.01, delta=1e-6, surgery={'name':'no_surgery', 'portion': 0.02, 'interval': 3})
    check_accuracy(_rnc.G, weight="weight", clustering_label="value", plot_cut=False)
    plt.show()
    show_results(_rnc.G)
    return G

if __name__ == "__main__":
    main()

