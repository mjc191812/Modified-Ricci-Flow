import networkx as nx

def create_sbm(sizes, p_intra, p_inter):
    # Calculate the number of communities
    num_communities = len(sizes)
    
    # Initialize the probability matrix
    probs = [[0] * num_communities for _ in range(num_communities)]
    
    # Fill the probability matrix
    for i in range(num_communities):
        for j in range(num_communities):
            if i == j:
                probs[i][j] = p_intra
            else:
                probs[i][j] = p_inter
    
    # Generate the random block model graph
    G = nx.stochastic_block_model(sizes, probs)
    
    return G


sizes = [250, 250]  # Size of each community
p_intra = 0.15  # Probability of an edge between two nodes within the same community
p_inter = 0.05  # Probability of an edge between two nodes in different communities


p_inter_values = [i / 100 for i in range(1, 11)]
for p_inter in p_inter_values:
    G = create_sbm(sizes, p_intra, p_inter)
    # Save the result to the desired path.


community_number_values = [2,3,4,5,6,7,8]
for community_number in community_number_values:
    sizes = [250] * community_number
    G = create_sbm(sizes, p_intra, p_inter)
    # Save the result to the desired path.
