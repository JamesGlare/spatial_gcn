from typing import List, Any, Dict
import numpy as np
from numpy import linalg as LA
from typing import Tuple, Union, Type, Optional, Dict, List, Iterable, Set, Callable

def check_symmetric(adj : np.ndarray, tol : float = 1e-8) -> bool:
    """
        Aux function used to find if an adjacency
        matrix is symmetric (and the underlying graph thus undirected).
    """
    return np.all(np.abs(adj-adj.T) < tol)

class SpatialGraph:
    """
        In this random graph, nodes have positions and a are more likely
        to establish links with nearby nodes. The probability to establish a link
        decreases as specified by a spatial kernel.
    """
    class Node:
        """
            Node class.
            Contains the node information of the graph.
        """
        def __init__(self,  graph : 'SpatialGraph',
                            coords : np.ndarray) -> None :
            self.id : int                = graph._node_count
            self.neighbours : Dict[int, 'Node'] = dict() # id -> Node
            graph._node_count += 1
            self.coords = coords 

        def __eq__(self, other : 'Node' ) -> bool :
            if isinstance(other, self.__class__):
                return self.id == other.id
            else:
                return False

        def has_edge_to(self, idx : int) -> bool :
            if idx == self.id:
                return True
            else:
                return idx in self.neighbours.keys()
        
        def __repr__(self) -> str :
            return str(self.id)
         
        def __iter__(self) -> Iterable[int] :
            self.n = 0
            self.key_list = list(self.neighbours.keys())
            while self.n < len(self.neighbours):
                retval = self.key_list[self.n]
                self.n += 1
                yield retval
    
        def pos(self) -> Tuple[float] :
            return tuple(self.coords)

    def __init__(self, 
                 num_nodes : int, 
                 dim : int = 2, 
                 directed : bool = False                 
                 ) -> None :
        self.nodes : Dict[int, 'SpatialGraph.Node'] = dict()
        self.num_nodes = num_nodes
        self.dim = dim
        self.directed = directed
        self._node_count : int = 0
       
    @classmethod
    def at_random(cls,  num_nodes : int,
                        dim : int,
                        edge_kernel : Callable[[np.ndarray,np.ndarray], np.ndarray],
                        directed : bool = False) -> 'SpatialGraph':
        graph = cls(num_nodes = num_nodes,
                    dim = dim,
                    directed = directed)
        graph._populate(None)
        coordinates = graph.get_coords()
        build_edge_to : Callable[['SpatialGraph.Node'], List[int]] = lambda node : np.argwhere( 
                                        np.random.rand( num_nodes ) <  edge_kernel(node.coords, graph.get_coords()) 
                                        ).flatten().tolist()
        graph._build_edges(build_edge_to)
        return graph

    @classmethod
    def from_adjacency_and_coordinates(cls, adj : np.ndarray, 
                                            coordinates : np.ndarray
                                        ) -> 'SpatialGraph':
        if len(adj.shape) <  1:
            raise ValueError("Adjancency matrix has to be two-dimensional (with opt. batch dimension) and has to have elements in it, but is {}-shaped.".format(adj.shape))
        elif not adj.shape[-2] == adj.shape[-1]:
            raise ValueError("Adjancecy matrix has to be a square matrix, but is {}-shaped.".format(adj.shape))
        if len(coordinates.shape) < 1:
            raise ValueError("Coordinates need to be at least 2-dimensional array, but are {}-shaped.".format(coordinates.shape))

        num_nodes = int(adj.shape[0])
        dim = int(coordinates.shape[-1])
        directed : bool = not check_symmetric(adj)
        graph = cls(num_nodes = num_nodes, 
                    dim = dim,
                    directed = directed)
        graph._populate(coordinates)
        build_edge_to : Callable[['SpatialGraph.Node'], List[int]] = lambda node : np.argwhere( adj[node.id]  == 1).flatten().tolist()
        graph._build_edges(build_edge_to)
        return graph

    def _populate(self, coordinates : Optional[np.ndarray]) -> None:
        """
            Populate the graph with unconnected nodes.
            Randomized initialization if no coordinates are passed.
            Coordinates have to be of shape [N, dim]
        """ 
        if coordinates is not None\
        and not tuple(coordinates.shape) == (self.num_nodes, self.dim):
            raise ValueError("_populate - shape error. Expected (num_nodes, dim), but got {}.".format(coordinates.shape)) 
        coordinates = coordinates if coordinates is not None else np.random.rand(self.num_nodes, self.dim)

        for node_coords in coordinates: # iterate over first dimension
            new_node = SpatialGraph.Node(self, node_coords)
            self.nodes[new_node.id] = new_node

    def _build_edges(self, build_edge_to : Callable[['SpatialGraph.Node'], List[int]]) -> None:
        """
            Iterate over all possible (n over 2) edges and build nodes.
            If no adjacency matrix is provided, build edges according to distance-law.
        """
        for node in self.nodes.values():
            for idx in build_edge_to(node):
                if not node.has_edge_to(idx) \
                and not idx == node.id: # prevent self-links
                    node.neighbours[idx] = self.nodes[idx]
                    # put reverse edge ?
                    if not self.directed:
                        self.nodes[idx].neighbours[node.id] = node

    def edges(self) -> Set[Tuple[int, int]] :
        """
            This function should return a set of tuple of ids (id_0, id_1).
            id_0 identifies the 'from'-node, while id_1 the 'to'-node represents.
        """  
        edges : Set[Tuple[int, int]] = set()
        for node_id in self.nodes: # iterator over id's
            for adj_node in self.nodes[node_id]:
                edge = (node_id, adj_node)
                if self.directed:
                    edges.add(edge)
                else:
                    if edge[::-1] not in edges: # if reverse edge not in edges...
                        edges.add(edge)
        return edges
    
    def __len__(self) -> int :
        return self.num_nodes

    def __str__(self) -> str :
        repr_str_list = []
        for node in self.nodes.values():
            id_list = map(str, node.neighbours.keys())
            repr_str_list.append(str(node) + " -> " + ",".join(id_list))
        return "\n".join(repr_str_list)

    def __getitem__(self, id : int) -> 'SpatialGraph.Node' :
        """
            Returns node at index id.
        """
        if id not in self.nodes.keys():
            raise IndexError("Index is out of bounds.")
        return self.nodes[id]
    
    def get_coords(self) -> np.ndarray :
        coords = np.zeros((self.num_nodes,self.dim))
        for node in self.nodes.values():
            coords[node.id, :] = node.coords
        return coords
"""
    -------------------------------------------------------------------------------------------
"""
class ErdosRenyiGraph:
    """
        Simple implementation of an Erdős–Rényi graph managing a web of nodes.
    """
    """
        -------------------------------------------------------------------------------------------
    """
    class Node:
        total_node_num = 0

        def __init__(self):
            # (1) find coordinates
            self.dim = 2
            self.coords = np.random.rand(self.dim)
            self.id = ErdosRenyiGraph.Node.total_node_num # should be unique
            ErdosRenyiGraph.Node.total_node_num +=1 
            self.edges = dict() # dict key: id of other node

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.id == other.id
            else:
                return False

        def has_edge_to(self, idx):
            if idx == self.id:
                return True
            else:
                return idx in self.edges.keys()
        
        def __repr__(self):
            return str(Node.id) + " "+ str(Node.coords)
    """
        -------------------------------------------------------------------------------------------
    """
    def __init__(self, num_nodes, edge_prob = 0.2):
        self.nodes = dict()
        self.edge_prob = edge_prob
        #1 populate the graph with unconnected nodes 
        for _ in range(num_nodes):
            new_node = ErdosRenyiGraph.Node()
            self.nodes[new_node.id] = new_node

        #2 Iterate over all possible (n over 2) edges and build nodes
        for node in self.nodes.values():
            curr_id = node.id
            build_edge_to = np.argwhere(np.random.rand(num_nodes ) < edge_prob).flatten().tolist() # elements in range 0, num_nodes -1
            for idx in build_edge_to:
                if not node.has_edge_to(idx):
                    node.edges[idx] = self.nodes[idx]

    def __str__(self):
        repr_str = ''
        for node in self.nodes.values():
            repr_str += str(node) + " -> "
            repr_str += str(node.edges.keys()) + "\n"
        return repr_str
    
    def __getitem__(self, id):
        """
            Returns node at index id.
        """
        if id not in self.nodes.keys():
            raise ValueError("Index is out of bounds.")
        return self.nodes[id]