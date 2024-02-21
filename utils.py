import torch
import torch_geometric
import numpy as np


def get_all_incoming_neighbours_of_node(node_id, edges, train_ids=None):
    """
    returns a tensor with all neighbours of a node (index)
    @param node_id: ID of the node you wnat to get all neighbours from
    @param edges: torch [[],[]] with edges
    
    returns tensor with neighbour node IDs
    """
    # if train_ids is not None:
    #     edges = torch_geometric.utils.subgraph(train_ids, edges)[0]
    #     return edges[0][torch.where(edges[1]==node_id)]
    return edges[0][torch.where(edges[1]==node_id)]


def get_matrix_indicating_same_labels(labels:torch.tensor):
    """
        This function returns a binary matrix indicating which nodes have the same labels
        a_ij = 0: nodes a_i and a_j don't have the same label
        a_ij = 1: ndoes a_i and a_j have the same label
    """
    return (torch.cdist(labels.unsqueeze(-1).float(), labels.unsqueeze(-1).float())==0).int()


def get_matrix_indicating_different_labels(labels:torch.tensor):
    """
        This function returns a binary matrix indicating which nodes have different labels
        a_ij = 0: nodes a_i and a_j have the same label
        a_ij = 1: ndoes a_i and a_j don't have the same label
    """
    return (torch.cdist(labels.unsqueeze(-1).float(), labels.unsqueeze(-1).float())!=0).int()
    

def get_number_conn_same_label_different_label(adj:torch.tensor, labels:torch.tensor, train_mask:torch.tensor=None, device="cpu"):
    """
        Returns two vectors indicating the number of connections to the same label and to different labels
        for each node in the graph
        adj: adjacency matrix (torch.tensor of shape [x,x])
        labels: label vector (torch.tensor of shape [x])
        train_mask: in case of transductive learning, you can pass a train mask and only the GNA of the training nodes will be evaluated
    """
    device = labels.device.type
    
    if train_mask is None:
        train_mask = torch.ones(adj.shape).to(device)
    else: 
        train_mask = torch.matmul(train_mask.unsqueeze(0).double().T, train_mask.unsqueeze(0).double()).int().to(device)
    matrix_ind_same_labels = get_matrix_indicating_same_labels(labels)
    matrix_ind_diff_labels = get_matrix_indicating_different_labels(labels)
    conn_same_labels = adj * matrix_ind_same_labels * train_mask
    conn_diff_labels = adj * matrix_ind_diff_labels * train_mask

    conn_same_labels = conn_same_labels.sum(dim=0)
    conn_diff_labels = conn_diff_labels.sum(dim=0)

    no_neighbours_in_subset = (conn_same_labels + conn_diff_labels)==0

    return conn_same_labels, conn_diff_labels, no_neighbours_in_subset