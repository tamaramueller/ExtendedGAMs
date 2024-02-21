# %%
import torch_geometric
import torch
import numpy as np
import extended_graph_metrics as gams
import networkx as nx


# %%
# define discrete/unweighted graph
graph_discrete = torch_geometric.data.Data(x=torch.tensor([[1, 2], [3, 1], [5, 5], [7, 2], [2,2], [3,2]], dtype=torch.float), edge_index=torch.tensor([[0, 1, 1, 2, 3, 1,3,0, 4, 5, 5,3,3], [1, 0, 2, 1, 2,3,1,3, 0, 2, 3,5,4]], dtype=torch.long))

labels_classification = torch.tensor([1,1,2,0,2,1], dtype=torch.long)
labels_regression = torch.tensor([[1.07], [2.10],[3.01],[5.44],[3.43],[0.48]], dtype=torch.float)

# %%
# plot graph
networkx_graph = torch_geometric.utils.to_networkx(graph_discrete)
nx.draw(networkx_graph, with_labels=True)
# %%
# define adjacency matrix of a continuous/weighted graph
adj_matrix_continuous = torch.tensor([[0.5, 1, 0.1, 0.2,0.2,0.1], [0.6, 0.4, 0.0, 0,0.5,0.1], [0, 0.7, 0, 0.1,0,0.1], [0, 0, 0.1, 0,0.4,0.2], [0.0,0.2,0.4,0.1,0.0,0.3], [0.5,0.6,0.2,0.1,0.,0.]], dtype=torch.float)

# %%
# calculate CCNS of discrete graph
ccns_discrete = gams.get_cross_class_neighbourhood_similarity_discrete(graph_discrete.edge_index, labels_classification, one_hot_input=False)
print("CCNS Matrix: ", ccns_discrete)
# %%
# calculate CCNS distance
ccns_distance = gams.get_ccns_distance(ccns_discrete)
print("CCNS distance: ", ccns_distance.item())
# %%
# calculate homophily of discrete graph
node_homophily_discrete, nr_same_labelled_neighbours, nr_diff_labelled_neighbours = gams.get_homophily_discrete(graph_discrete.edge_index, 6, labels_classification)
print("Mean node homophily of the graph: ", node_homophily_discrete.mean().item())
# %%
# calculate homphily for continuous graph
homophily_regression, nodewise_weights_same_label, nodewise_weights_diff_label = gams.get_homophily_continuous(adj_matrix_continuous, labels_classification)
print("Mean node homophily for a continuous adjacency matrix: " , homophily_regression.mean().item())
# %%
# calculate node homophily for regression on discrete graphs 
homophily_regression_discrete_mean, homophily_regression_discrete_std = gams.get_homophily_discrete_regression(graph_discrete.edge_index, 6, labels_regression)
print("Mean node homophily for a discrete adjacency matrix for regression: ", homophily_regression_discrete_mean.item())

# %%
# calculate node homophily for regression on continuous graphs
homophily_regression_continuous_mean, homophily_regression_continuous_std = gams.get_homophily_continuous_regression(adj_matrix_continuous, labels_regression)
print("Mean node homophily for a continuous adjacency matrix for regression: ", homophily_regression_continuous_mean)
# %%
