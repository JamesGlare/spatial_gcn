from typing import List, Tuple, Dict, Type, Union, Optional, Any, Callable
from graph import SpatialGraph
import numpy as np
import scipy.sparse as sp
from math import ceil
from torch.nn import MSELoss
import torch as T
import plotly.graph_objects as go
import warnings

def graph_plot( graph1 : SpatialGraph,
                graph2 : Optional[SpatialGraph] = None) -> None:
    def _create_edge_trace( graph : SpatialGraph, 
                            color : str = '#888') -> go.Scatter:
        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0, y0 = graph.nodes[edge[0]].pos()
            x1, y1 = graph.nodes[edge[1]].pos()
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color=color),
            hoverinfo='none',
            mode='lines')
        return edge_trace 

    def _create_node_trace( graph : SpatialGraph, 
                            colorscale : str  ='YlGnBu') -> go.Scatter:
        node_x = []
        node_y = []
        for node in graph.nodes.values():
            x, y = node.pos()
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale=colorscale,
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        return node_trace

    def _color_nodes(graph : SpatialGraph, node_trace : go.Scatter) -> None:
        node_adjacencies = []
        node_text = []
        for nr, node in enumerate(graph.nodes.values()):
            node_adjacencies.append(len(node.neighbours))

        node_trace.marker.color = node_adjacencies

    edge_trace1 = _create_edge_trace(graph1)

    node_trace1 = _create_node_trace(graph1, colorscale='Blues')
    _color_nodes(graph1, node_trace1)

    fig = go.Figure(data=[edge_trace1, node_trace1],
             layout=go.Layout(
                title='<br>SpatialGraph Plot',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    if graph2 is not None:
        edge_trace2 = _create_edge_trace(graph2, color='#aaa')
        node_trace2 = _create_node_trace(graph2, colorscale='Reds')
        _color_nodes(graph2, node_trace2,)
        fig.add_trace(edge_trace2)
        fig.add_trace(node_trace2)
        
    fig.show()

def to_np(tensor : T.tensor) -> np.ndarray:
    return tensor.clone().detach().cpu().numpy()

def graph_probability(  graph : SpatialGraph,
                        kernel : Callable[[np.ndarray,np.ndarray], np.ndarray]
                        ) -> float:
    """
        For a given kernel, compute how probable a certain graph is.
        In its current form, the function assumes a uniform distribution
        of graph nodes across the interval.
    """
    prob : float = 1.0
    edge_set : Set[Tuple[int, int]] = set() # in order to prevent double counting of edges in undirected graphs we have to some bookkeeping
    for edge in graph.edges():
        if edge not in edge_set:
            from_node = graph[edge[0]]
            to_node = graph[edge[1]] 
            kernel_prob = float(kernel(from_node.coords, to_node.coords))
            prob *= kernel_prob # edge probs are independent
            ## store edge in set in reverse order
            edge_set.add(edge[::-1])

    return prob

def adjacency(  graph : SpatialGraph, 
                normalize : bool = True,
                sparse : bool = False
                ) -> np.ndarray :
    """
        Only supports undirected graphs at the moment.
    """
    if graph.directed:
        raise NotImplementedError("Directed graphs are currently not supported.")
    dtype = np.float if normalize else np.int

    adj = np.zeros((graph.num_nodes, graph.num_nodes), dtype=dtype)
    if sparse:
        adj = sp.coo_matrix(adj)
    for node in graph.nodes.values():
        for adj_node in node.neighbours.values():
            adj[node.id, adj_node.id] = 1
    return normalize_adj(adj, sparse) if normalize else adj

def normalize_adj(  adj : np.ndarray, 
                    sparse : bool = False
                    ) -> Union[np.ndarray, sp.spmatrix]:
    """ From T Kipf's code
        Symmetrically normalize adjacency matrix."""
    if sparse:
        adj = sp.coo_matrix(adj)                    # [N,N]
    rowsum = np.array(adj.sum(1))                   # [N,]
    
    d_inv_sqrt = np.power(rowsum, -0.5)             # [N,], may issue runtime warnings (div by zero)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.           # []
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) if sparse else np.diag(d_inv_sqrt)   #[N,N]
    
    if sparse:
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    else:
        return ((adj @ d_mat_inv_sqrt).transpose() @ d_mat_inv_sqrt) # not quite sure why this order = D^T A^T D, D^T = D, A^T = A - the transpose is unncessary?!

def dist(x : np.ndarray, other_x : np.ndarray) -> np.ndarray :
    #return LA.norm(coords - x)
    sq_err = (x-other_x)**2
    return np.sqrt(np.sum(sq_err, axis=-1))

def dist_to_others( node : Union[np.ndarray, SpatialGraph.Node], 
                    other : Union[np.ndarray, SpatialGraph]
                        ) -> np.ndarray:
    if isinstance(node, SpatialGraph.Node):
        node_coords = node.coords
    else:
        node_coords = node
    
    if isinstance(other, SpatialGraph):
        other_coords = other.get_coords()
    else:
        other_coords = other
    return dist(node_coords, other_coords) # use broadcasting

def create_adjacencies(graphs : List[SpatialGraph], normalize : bool = True ) -> np.ndarray:
    """
        Creates batched tensor containing all adjacency matrices of the graphs.
        Note that normalization also adds the nxn-identity to include node self-links.
    """
    dtype = np.float if normalize else np.int
    return np.array([adjacency(graph, normalize) for graph in graphs], dtype=dtype)

def create_labels(graphs : List[SpatialGraph] ) -> np.ndarray:
    """
        Function accepts a list of graphs, each with N nodes and p parameters.
        The number of nodes and parameters should be constant throughout the list.
    """
    if not graphs: # None or empty
        return np.array([])

    N = graphs[0].num_nodes
    p = graphs[0].dim
    n_batch = len(graphs)

    labels = np.zeros((n_batch, N, p))
    
    for i in range(n_batch):
        labels[i,:,:] = graphs[i].get_coords()

    return labels

def get_mask_indices( n_batch : int,
                      N : int,
                      n_mask : int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
        Use numpy advanced indexing to mask coordinates.
        Return value is numpy integer array tuple, each matrix of shape [n_batch, n_mask].
    """
    rows = np.repeat(np.arange(n_batch)[:,None], n_mask, axis=1)
    cols =  np.sort(np.array([np.random.choice(np.arange(N), size=n_mask, replace=False) for _ in range(n_batch)], dtype=np.int), axis=1)
    return rows, cols # Both are shape [n_batch, n_mask]

def create_mask(n_batch : int,
                N : int,
                n_par : int = 2,
                masked_proportion : float = 0.1) -> np.ndarray:
    

    mask = np.random.binomial(1, 1-masked_proportion, size=(n_batch, N))
    mask = np.repeat(mask[:,:,None], n_par, axis = 2)
    mask = mask/np.mean(mask)
    return mask

def masked_mse_loss(prediction : Union[T.tensor, np.ndarray],
                    groundtruth : Union[T.tensor, np.ndarray],
                    masked_proportion : float = 0.1) -> Union[T.tensor, np.ndarray]:
    """
        Implements the MSE loss for positional coordinates.
    """
    n_batch, N, n_par = groundtruth.shape
    mask = create_mask(n_batch, N, n_par, masked_proportion)
    if not isinstance(prediction, np.ndarray):
        mask = T.tensor(mask, 
                    dtype=T.float32, 
                    requires_grad=False)
    
    loss_matrix = 0.5*(groundtruth - prediction)**2
    return (mask*loss_matrix).mean()

def compute_distance_histogram(graph : SpatialGraph) -> List[float] :
    distances = []
    for node in graph.nodes.values():
        distances.extend([dist(node.coords, graph.nodes[key].coords) for key in node.neighbours.keys() ])
    #hist = np.histogram(distances, bins=10)
    return distances

def laplacian(  graph : SpatialGraph, 
                sparse : bool = False
                ) -> Union[np.ndarray, sp.spmatrix] :
    """
        Only supports undirected graphs.
    """
    adj = adjacency(graph, sparse=sparse)
    dgr =  sp.diags(np.array(adj.sum(1))) if sparse else np.diag(np.array(adj.sum(1)))
    return adj - dgr