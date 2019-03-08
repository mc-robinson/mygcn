# mygcn
  
This repo contains my own implementation of a *very simple* graph convolutional network (gcn) using [PyTorch](https://pytorch.org/) and [DGL](https://www.dgl.ai/) (DeepGraphLibrary).

The code is largely adapted from the [pygcn](https://github.com/tkipf/pygcn) implementation of Kipf and Welling and the helpful [DGL tutorials](https://docs.dgl.ai/tutorials/models/index.html). The background for the models is nicely explained in their [accompanying blog post](https://tkipf.github.io/graph-convolutional-networks/). 

The necessary dependencies are specified in the `env-files/` folder, which contains the `.yml` files for creating [conda](https://www.anaconda.com/) environments for easy portability. Unfortunately, [DeepChem's](https://DeepChem.io) latest installation was not working properly, and thus an older distribution had to be used. Therefore, the code with DeepChem dependencies must be run in a separate environment from the rest of the code relying on PyTorch and DGL.

For understanding the basic usage of the code, I highly recommend the `notebooks/` folder, which contains simple tests/benchmarks of the models and shows basic usage. 

Finally, It should be noted that this implementation is just a proof of concept and does not produce state of the art results. Implementations like [that of DeepChem](https://deepchem.io/docs/notebooks/graph_convolutional_networks_for_tox21.html) are highly optimized and more robust. However, this implementation is quite simple to understand due to the use of DGL, and is perhaps a good starting place for understanding a gcn for chemistry from scratch.

-- Matt Robinson
