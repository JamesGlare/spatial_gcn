import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
from typing import List, Type, Optional, Any, Iterable, Union
from utils import masked_mse_loss, to_np
"""
    Utility functions
"""
def _matrix(n_left : int, 
            n_right : int, 
            dtype : T.dtype = T.float32,
            xavier : bool = True,
            requires_grad : bool = True ) -> nn.Parameter:
    matrix = T.randn(n_left, 
                     n_right, 
                     dtype = dtype)
    if xavier:
        nn.init.xavier_uniform_(matrix, 
                    gain = nn.init.calculate_gain('relu'))

    return nn.Parameter(matrix, requires_grad = requires_grad)

def _bias(  n : int,
            dtype : T.dtype = T.float32,
            requires_grad : bool = True) -> nn.Parameter:
    bias = T.randn(n, dtype = dtype)
    nn.init.zeros_(bias)
    return nn.Parameter(bias, requires_grad = requires_grad)

class GCN(nn.Module):
    """
        Graph Convolutional Networks
        The model is built for a transductive,
        semi-supervised classification/regression of homophilic
        graph node labels.
        What does this mean?
        It means that we assume here that we know the graph structure,
        i.e. we know all nodes and all edges (adjacency matrix).
        Also, we assume that this knowledge is available immediately from the getgo
        and doesn't come bit-by-bit like in a stream for instance.
        
        Importantly, (it's semi-supervised!) we know a few node labels in the beginning.

        Furthermore, we assume that graph nodes are connected to nodes that
        are similar to them, i.e. nodes are more likely to be connected to similar nodes.
        This property is known as homophilia. Our GCN has to deduce the label from 
        labels of neighbours and knowledge of their labels.

        Learn:
        [1] https://en.wikipedia.org/wiki/Transduction_(machine_learning)
        [2] 
    """
    class Layer(nn.Module):
        """
            Graph convolutional layer.
        """
        ## Set these parameters when initializing GCN
        n_nodes : Optional[int] = None
        n_par :  Optional[int] = None
    
        def __init__(self,  n_left : int,
                            n_right : int ) -> None:
            super(GCN.Layer, self).__init__()
            """
                Operation:
                A.X.W + b 
                    A: Adjacency matrix
                    X: Input
                    b: bias terms

                Cardinalities:
                A [b,n,n], X [b,n,p] -> W [p,h], b [h,]
            """
            self.W = _matrix(n_left = n_left, n_right = n_right, xavier=True)
            self.b = _bias(n = n_right)

        def forward(self, A : T.tensor, X : T.tensor) -> T.tensor:
            return A @ X @ self.W + self.b
        
    device = 'cuda' if T.cuda.is_available() else 'cpu'
        
    def __init__(self,  n_nodes : int,
                        n_par : int, # should be equal to node number
                        list_hidden : List[int] = [50,50],
                        act = F.relu,
                        batch_norm : bool = False) ->  None:
        super(GCN, self).__init__()
        """
            Initialize a GCN.
            The GCN weight matrices are independent of the number of nodes in the graph.
            The normalization layers need to know however.
        """
        n_layers : int = len(list_hidden) # equals number layers - 1 
        self.layers = nn.ModuleList()
        self.normalizations = nn.ModuleList()

        for i,n_hidden in enumerate(list_hidden):
            """
            Example: 2 hidden Layers with list_hidden = [n_h0]
                i == 0: W [n_par, n_h0]
                i == 1: W [n_h0, n_par]
                A(A.X.W0).W1 -> [n,n].([n,n].[n,p].[p,h0]).[h0,p] -> [n,p]
            """
            n_left  : int = n_par if i == 0 else list_hidden[i-1]
            n_right : int = list_hidden[i]
            self.layers.append(GCN.Layer( n_left = n_left, 
                                          n_right = n_right))
            if batch_norm:
                self.normalizations.append(nn.BatchNorm1d(n_nodes))
            else:
                self.normalizations.append(nn.Identity())

        ## Append a last layer which brings dimensions back to [n_par] on the right
        n_par_last : int = list_hidden[-1] if n_layers > 0 else n_par
        self.layers.append(GCN.Layer(n_left = n_par_last, n_right = n_par))
        self.normalizations.append(nn.Identity())
        
        ## Collect activation functions
        self.acts : Callable[T.tensor, T.tensor] = [T.relu]*n_layers + [lambda x: x] # the last layer does not apply any non-linearity

    def to_device(self):
        self.to(GCN.device) # TODO: does this affect the moduleList etc ?

    def forward(self, A : Union[np.ndarray, T.tensor], # [n_batch, n_nodes, n_nodes]
                      X : Union[np.ndarray, T.tensor]  # [n_batch, n_nodes, n_par  ]
                      ) -> T.tensor:
        """
            Predict coordinates/classes of graph with 
                Adjacency matrix A and
                masked coordinates X
            
            Function accepts either Torch.Tensors or numpy arrays,
            the former are assumed to have been moved to the correct device, GCN.device.
        """
        if isinstance(X, np.ndarray):
            X : T.tensor = T.tensor(X, 
                                device = GCN.device,
                                dtype = T.float32, 
                                requires_grad = True) # batched coordinates [n_batch, n_nodes, n_par ]
        
        if isinstance(A, np.ndarray):
            A : T.tensor  = T.tensor(A, 
                                device = GCN.device,
                                dtype = T.float32, 
                                requires_grad = False) # batched adjacencies [n_batch, n_nodes, n_nodes]
        
        for act, norm, layer in zip(self.acts, self.normalizations, self.layers):
            X =  act(norm(layer(A, X)))
            
        return X

    def evaluate(self,  A :  Union[np.ndarray, T.tensor], # [n_batch, n_nodes, n_nodes]
                        X :  Union[np.ndarray, T.tensor]  # [n_batch, n_nodes, n_par  ] 
                        ) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X : T.tensor = T.tensor(X, 
                                device = GCN.device,
                                dtype = T.float32, 
                                requires_grad = True).to(GCN.device) # batched coordinates [n_batch, n_nodes, n_par ]
        if isinstance(A, np.ndarray):    
            A : T.tensor  = T.tensor(A, 
                                device = GCN.device,
                                dtype = T.float32, # integer -> fp for float matrix multiplication
                                requires_grad = False).to(GCN.device) # batched adjacencies [n_batch, n_nodes, n_nodes]
        return to_np(self(A,X))
class GraphOptimizer:
    """
        Implements optimization and loss for GraphConvolution operation.
    """
    def __init__(self,  parameters :  Iterable[nn.Module],
                        lr : float ) -> None:
        ## Initialize the optimizer after parameters are known
        self.lr : float = lr
        self.optimizer = Adam(parameters, lr=lr)

    def step(self,  X : Union[np.ndarray, T.tensor], # Groundtruth         
                    X_HAT : Union[np.ndarray, T.tensor], # Prediction
                    mask_proportion : float = 0.9, # 
                    training : bool = True ) -> float:
        """
            Wrapper for the multiple substeps involved in a
            training step.
        """
        if isinstance(X, np.ndarray):
            X : T.tensor = T.tensor(X, 
                                dtype = T.float32, 
                                requires_grad = True).to(GCN.device) # batched coordinates [n_batch, n_nodes, n_par ]
        if isinstance(X_HAT, np.ndarray):
            X_HAT : T.tensor  = T.tensor(X_HAT, 
                                dtype = T.float32, 
                                requires_grad = False).to(GCN.device) # batched adjacencies [n_batch, n_nodes, n_nodes]

        loss : T.tensor = masked_mse_loss(X, X_HAT, mask_proportion)
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return float(loss.data)

    def loss(self,  X : Union[np.ndarray, T.tensor],     # Groundtruth         
                    X_HAT : Union[np.ndarray, T.tensor], # Prediction
                    mask_proportion : float = 0.9
                    ) -> float:
        loss : Union[np.ndarray, T.tensor] = masked_mse_loss(X, X_HAT, mask_proportion)
        return float(loss)