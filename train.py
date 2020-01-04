from graph import ErdosRenyiGraph, SpatialGraph
from os.path import join, exists
from os import getcwd
import sys
import matplotlib.pyplot as plt
import argparse
from utils import adjacency, normalize_adj, graph_plot, create_labels, create_adjacencies, dist, to_np
from typing import List, Dict, Any, Tuple, Iterator, Callable
import numpy as np
from model import GCN, GraphOptimizer
import torch as T
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_args(argv : List[str]) -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', 
                        default = 3000,
                        type=int,
                        required=False)
    parser.add_argument('--n_test', 
                        default = 200,
                        type=int,
                        required=False)
    parser.add_argument('--n_batch', 
                        default = 15,
                        type=int,
                        required=False)   
    parser.add_argument('--n_nodes', 
                        default = 1000,
                        type=int,
                        required=False)   
    parser.add_argument('--n_par', 
                        default = 2,
                        type=int,
                        required=False)                    
    parser.add_argument('--n_eyeball', 
                        default = 5,
                        type=int,
                        required=False)
    parser.add_argument('--seed', 
                        default = 42,
                        type=int,
                        required=False)
    parser.add_argument('--lr', 
                        default = 5e-4,
                        type=float,
                        required=False)
    parser.add_argument('--mask_proportion', 
                        default = 0.7,
                        type=float,
                        required=False)
    parser.add_argument('--edge_prob', 
                        default = 0.8,
                        type=float,
                        required=False)
    parser.add_argument('--alpha', 
                        default = 40,
                        type=float,
                        required=False)
    parser.add_argument('--model_file_name', 
                        default = "GCN.pt",
                        type=str,
                        required=False)
    parser.add_argument('--load_model', 
                        action='store_true',
                        required=False)
    parser.add_argument('--print_steps',
                        default = 100,
                        type=int,
                        required=False)
    args = parser.parse_args()
    return args

def batch_list( to_batch : List[Any], 
                batch_size : int
                ) -> Iterator[Any]:
    """
        Generator for simple batch-wise loading of data.
    """
    for i in range(0,len(to_batch), batch_size):
        yield to_batch[i:i+batch_size]

def power_law_kernel(alpha : float, 
                     edge_prob : float
                    ) -> Callable[[np.ndarray,np.ndarray], np.ndarray]:
        """
            The probability of an edge between two nodes a,b scales with
            p(a,b) ~ 1/(1+d)^alpha with d being the euclidean distance between the node coordinates.
        """
        return lambda node_coords, all_coords : edge_prob*np.reciprocal(np.power( 1 + dist(node_coords, all_coords), alpha)) # 1/(1+d)^alpha  

def plot_coords(    coords1 : np.ndarray,
                    coords2 : np.ndarray
                    ) -> None:
    """
        Helper function to print graph node coordinates.
    """
    fig,ax = plt.subplots()
    plt.scatter(coords1[:,0], coords1[:,1], marker='x')
    plt.scatter(coords2[:,0], coords2[:,1], marker='o')
    
    plt.show()

def plot_example(   model : GCN, 
                    n_nodes: int,
                    n_par : int,
                    kernel : Callable[[np.ndarray,np.ndarray], np.ndarray]
                ) -> None:
    example = SpatialGraph.at_random( num_nodes = n_nodes, 
                                      dim = n_par,
                                      edge_kernel = kernel )

    X : np.ndarray = create_labels([example]) # batched coordinates [1, n_nodes, n_par]
    A : np.ndarray = create_adjacencies([example]) # batched adjacency matrices [1, n_nodes, n_nodes]
    X_HAT = model.evaluate(A, X)
    
    A_unorm : np.ndarray = create_adjacencies([example], normalize=False) 
    prediction = [SpatialGraph.from_adjacency_and_coordinates(adj, coords) for adj, coords in zip(A_unorm,X_HAT)] # unbatch 
    graph_plot(example, prediction[0])
    #plot_coords(X[0], X_HAT[0])

def main(argv : List[str]) -> None:

    args = parse_args(argv)
    print('Command Line Arguments: ------')
    [print(arg,':',getattr(args, arg)) for arg in vars(args)]
    print('------------------------------')
    """
        Parameters of training & evaluation ---------------------------------------------------------------------------
    """
    n_train : int = args.n_train
    n_batch : int = args.n_batch
    n_nodes : int = args.n_nodes
    n_par   : int = args.n_par

    mask_proportion : float = args.mask_proportion
    lr : float = args.lr

    ## set seed
    np.random.seed(args.seed)
    T.manual_seed(args.seed)

    ## supress numpy warnings which may occur in normalize_adj(...)
    np.seterr(divide='ignore')
    
    """
        Set up model & optimizer --------------------------------------------------------------------------------------
    """
    model : GCN = GCN(n_nodes = n_nodes, list_hidden=[50,100, 50], n_par = n_par, batch_norm = False)
    model.train()
    model_path : str = join(getcwd(), args.model_file_name)

    ## Setup optimizer
    gcn_optimizer : GraphOptimizer = GraphOptimizer(model.parameters(), lr = lr)
    model.to_device() # has to happen after optimizer instantiation

    ## Setup edge kernel
    kernel : Callable[[np.ndarray,np.ndarray], np.ndarray] = power_law_kernel(  edge_prob = args.edge_prob, 
                                                                                alpha = args.alpha)
    
    ## Setup cross validation kernel
    cv_kernel : Callable[[np.ndarray,np.ndarray], np.ndarray] = power_law_kernel(   edge_prob = min(1.0, args.edge_prob+0.1), 
                                                                                    alpha = args.alpha+5)
    
    ## Show example graph
    print("Plotting example graph...")
    graph_plot(SpatialGraph.at_random( num_nodes = n_nodes, 
                                       dim = n_par, 
                                       edge_kernel = kernel))
    
    print("Commencing training loop...")
    if not args.load_model:
        for step in tqdm(range(0, n_train, n_batch)):
            """
                Training loop -----------------------------------------------------------------------------------------
                No epochs used here since we can create infinitely many graphs.
                Batches of graphs are created on-demand within the loop to save memory.
            """
            graph_batch : List[SpatialGraph] = [SpatialGraph.at_random( # explicitly execute constructor n times
                                                    num_nodes = n_nodes, 
                                                    dim = n_par, 
                                                    edge_kernel = kernel) for _ in range(n_batch)] 

            X : np.ndarray = create_labels(graph_batch)         #  [n_batch, n_nodes, n_par]
            A : np.ndarray = create_adjacencies(graph_batch)    #  [n_batch, n_nodes, n_nodes]

            X_HAT : T.tensor = model(A,X) # predicted coordinates, [n_batch, n_nodes, n_par]
            loss : float = gcn_optimizer.step(X, X_HAT, mask_proportion)

            if step % args.print_steps == 0:
                ## Apply model to slightly differently distributed graph
                graph_batch : List[SpatialGraph] = [SpatialGraph.at_random( # explicitly execute constructor n times
                                                    num_nodes = n_nodes, 
                                                    dim = n_par, 
                                                    edge_kernel = cv_kernel) for _ in range(n_batch)] 
                X_cv : np.ndarray = create_labels(graph_batch)         #  [n_batch, n_nodes, n_par]
                A_cv : np.ndarray = create_adjacencies(graph_batch)    #  [n_batch, n_nodes, n_nodes]

                X_HAT_cv : T.tensor = model(A_cv,X_cv) # predicted coordinates, [n_batch, n_nodes, n_par]
                cv_loss : float = gcn_optimizer.step(X_cv, X_HAT_cv, mask_proportion)

                tqdm.write("Loss/CV loss at {} - {:.3f}/{:.3f}".format(step, loss, cv_loss))

        ## Save the model
        print("Saving the trained model at {} ...".format(model_path))
        T.save(model.state_dict(), model_path)
    else:
        if exists(model_path):
            print("Loading model from {} ...".format(model_path))
            model.load_state_dict(T.load(model_path))
        else:
            print("Model does not exist at {}. Please specify path to model file.".format(model_path))
            sys.exit()
    """
        Evaluate model on a test set ----------------------------------------------------------------------------------
    """
    avg_loss : float = 0.0
    model.eval()
    print("Commencing evaluation loop...")
    for _ in tqdm(range(0, args.n_test, n_batch)):
        graph_batch : List[SpatialGraph] = [SpatialGraph.at_random( # explicitly execute constructor n times
                                                num_nodes = n_nodes, 
                                                dim = n_par, 
                                                edge_kernel = kernel) for _ in range(n_batch)] 
        X : np.ndarray = create_labels(graph_batch)
        A : np.ndarray = create_adjacencies(graph_batch)

        X_HAT : np.ndarray = model.evaluate(A,X) # predicted coordinates
        loss : float = gcn_optimizer.loss(X, X_HAT, mask_proportion)
        avg_loss += loss
    
    loss_per_node = avg_loss /(n_nodes*args.n_test)
    avg_loss /= args.n_test
    print("Test set loss per node: {:.3f} \%".format(100*loss_per_node))
    print("Test set average loss: {:.3f} \%".format(100*avg_loss))
    
    ## Last step: Eyeball a few predictions and groundtruths
    print("Preparing graph visualization...")
    for _ in range(args.n_eyeball):
        plot_example(model, n_nodes, n_par, kernel)

if __name__ == "__main__":
    main(sys.argv)