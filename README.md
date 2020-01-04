# Graph Convolutional Network for semi-supervised spatial inference

I am building a simple example of a graph convolutional network to predict positions of nodes in a spatial graph.
The network is trained on homophilic spatial graphs (i.e. close-by nodes are more likely to be connected) that are partially occluded.

For this purpose, I wrote a simple graph class which supports edge-initialization from cumstom-defined spatial kernels.
While nodes are distributed uniformly across [0,1], the probability of an edge between any two nodes i,j is given by
p(i,j) ~ 1/(1+d)^alpha where d is the euclidean distance between the nodes while alpha denotes an exponent.

## What do I need to do to run it?

You need to have pytorch, numpy, tqdm, matplotlib and plotly installed (requirements.txt will follow).
Then simply run
`python train.py --n_train 3000 --n_test 200 --n_eyeball 5`

