'''
Code defining the various graph convolutional networks (GCN)

Much of the code here is borrowed from the implementations of:
    - DGL: https://docs.dgl.ai/tutorials/basics/4_batch.html
    - Kipf and Welling: https://github.com/tkipf/pygcn
    - Deepchem: https://arxiv.org/abs/1611.03199
    - Duvenaud: https://arxiv.org/abs/1509.09292

In particular, our approach uses a basic message passing framework to 
significantly simplify much of the code. There is no need for dealing with 
custom graph objects, as DGL efficiently handles the graph/message passing 
operations.

There are however disadvantages to our simple (likely overly so) approach.
In particular, the performance of our Duvenaud implentation suffers since we do
not construct a different weight matrix for atoms of a certian degree.
Futhermore, our featurization of this model is simplified; therefore, this model
should not be used for performance.

Our version of the DeepChem graph conv classifier performs significantly better 
than the Duvenaud implementation, yet will still suffer compared to the highly 
optimized official DeepChem implementation. This code should still be treated as
proof of concept and a simplified way to understand the network/implementation.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

# imports for typing
from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, \
                      Iterable, List, Mapping, NewType, Optional, Sequence, \
                      Tuple, TypeVar, Union                   
from types import SimpleNamespace

### MESSAGE PASSING FUNCTIONS: ###

def message(edge):
    "Generate a message to pass"
    msg = edge.src['h'] # note we do not include normalization as in paper
    return {'m': msg}

def sum_reduce(node):
    "Accumulate messages from neighboring nodes"
    accum = torch.sum(node.mailbox['m'], dim=1) # again note no norm
    return {'h': accum}

def mean_reduce(node):
    "Max over messages from neighboring nodes"
    accum = torch.mean(node.mailbox['m'], dim=1) # again note no norm
    return {'h': accum}

def max_reduce(node):
    "Max over messages from neighboring nodes"
    accum = torch.max(node.mailbox['m'], dim=1)[0] # again no norm
    return {'h': accum}

### LAYERS: ###
        
class GraphConvLayer(nn.Module):
    "generic GC layer, as defined by DGL and Kipf and Welling (w/o their norm)"
    def __init__(self, n_inputs:int, n_outputs:int,
                 activation:Callable):
        super(GraphConvLayer, self).__init__()
        # self.node_update = NodeApplyModule(n_inputs, n_outputs, activation)
        self.linear = nn.Linear(n_inputs, n_outputs)
        self.activation = activation

    def node_update(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}
        
    def forward(self, g, features):
        # initialize node features
        g.ndata['h'] = features
        g.update_all(message, mean_reduce)
        g.apply_nodes(func=self.node_update)
        return g.ndata.pop('h')

class DuvenaudGraphConvLayer(nn.Module):
    "As understood from https://arxiv.org/abs/1509.09292."
    def __init__(self, n_inputs:int, n_outputs:int,
                 activation:Callable=F.relu):
        super(DuvenaudGraphConvLayer, self).__init__()
        # note lack of degree specific weight matrices
        self.linear = nn.Linear(n_inputs, n_outputs)
        # not actually clear what their 'smooth function' is in paper
        self.activation = activation

    def node_update(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}
        
    def forward(self, g, features):
        # initialize node features
        g.ndata['h'] = features
        # our passing of edge features will not be the same as theirs.
        g.update_all(message, mean_reduce)
        g.apply_nodes(func=self.node_update)
        return g.ndata.pop('h')

class DuvenaudSoftmaxLayer(nn.Module):
    def __init__(self, n_inputs:int, n_outputs:int):
        super(DuvenaudSoftmaxLayer, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)
        self.activation = nn.Softmax(dim=1)

    def node_update(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}
        
    def forward(self, g, features):
        # initialize node features
        g.ndata['h'] = features
        g.apply_nodes(func=self.node_update)
        return g.ndata.pop('h')


class GraphPoolLayer(nn.Module):
    "Graph Pool layer to mimic that used in DeepChem"
    def __init__(self):
        super(GraphPoolLayer, self).__init__()
        
    def forward(self, g, features):
        # initialize node features
        g.ndata['h'] = features
        # note the simplicity of this given DGL library
        g.update_all(message, max_reduce)
        return g.ndata.pop('h')

class BatchNormLayer(nn.Module):
    "Batch Norm layer like that used in deepchem."
    def __init__(self, n_inputs:int):
        super(BatchNormLayer, self).__init__()
        self.bn = nn.BatchNorm1d(n_inputs)

    def nodewise_bn(self, node):
        h = self.bn(node.data['h'])
        return {'h': h}
        
    def forward(self, g, features):
        # initialize node features
        g.ndata['h'] = features
        g.apply_nodes(func=self.nodewise_bn)
        return g.ndata.pop('h')

### NETWORKS: ###

class DeepChemGCNRegressor(nn.Module):
    def __init__(self, 
                 n_inputs:int,
                 n_hidden:int=64, 
                 n_outputs:int=1,
                 n_hidden_layers:int=2, 
                 activation:Callable=F.relu,
                 dropout:float=0.0):

        super(DeepChemGCNRegressor, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(
            GraphConvLayer(n_inputs, n_hidden, activation=activation)
        ) 
        self.layers.append(
            BatchNormLayer(n_hidden)
        )
        self.layers.append(
            GraphPoolLayer()
        )
        # hidden layers
        for _ in range(n_hidden_layers-1):
            self.layers.append(
                GraphConvLayer(n_hidden, n_hidden, activation=activation)
            )
            self.layers.append(
                BatchNormLayer(n_hidden)
            )
            self.layers.append(
                GraphPoolLayer()
            )

        # dense layer to do pre graph gather step
        self.dense_layer = nn.Linear(n_hidden, 128)
        self.final_bn = nn.BatchNorm1d(128)
        # output layer, for pka prediction
        self.prediction_layer = nn.Linear(128, n_outputs)
        
    def forward(self, g):
        h = g.ndata['feat'] # this could be troublesome
        # I am skipping the averaging which could cause issues
        for idx, layer in enumerate(self.layers):
            if idx != 0:
                h = self.dropout(h)
            h = layer(g, h)

        # apply dense before graph gather    
        h = nn.ReLU()(self.dense_layer(h))
        # apply batch norm
        h = self.final_bn(h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h') # note Deepchem uses mean and max
        hg = nn.Tanh()(hg) # bit of a weird step used in DeepChem
        return self.prediction_layer(hg)

class DeepChemGCNClassifier(nn.Module):
    def __init__(self, 
                 n_inputs:int,
                 n_hidden:int=64, 
                 n_outputs:int=1,
                 n_hidden_layers:int=2, 
                 activation:Callable=F.relu,
                 dropout:float=0.0):

        super(DeepChemGCNClassifier, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(
            GraphConvLayer(n_inputs, n_hidden, activation=activation)
        ) 
        self.layers.append(
            BatchNormLayer(n_hidden)
        )
        self.layers.append(
            GraphPoolLayer()
        )
        # hidden layers
        for _ in range(n_hidden_layers-1):
            self.layers.append(
                GraphConvLayer(n_hidden, n_hidden, activation=activation)
            )
            self.layers.append(
                BatchNormLayer(n_hidden)
            )
            self.layers.append(
                GraphPoolLayer()
            )

        # dense layer to do pre graph gather step
        self.dense_layer = nn.Linear(n_hidden, 128)
        self.final_bn = nn.BatchNorm1d(128)
        self.classification_layer = nn.Linear(128, n_outputs)
        
    def forward(self, g):
        h = g.ndata['feat']
        for idx, layer in enumerate(self.layers):
            if idx != 0:
                h = self.dropout(h)
            h = layer(g, h)
        # apply dense before graph gather    
        h = nn.ReLU()(self.dense_layer(h))
        # apply batch norm
        h = self.final_bn(h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        hg = nn.Tanh()(hg)
        return self.classification_layer(hg)


class DuvenaudGCNClassifier(nn.Module):
    def __init__(self, 
                 n_inputs:int,
                 n_hidden:int=64, 
                 n_outputs:int=1,
                 n_hidden_layers:int=2, 
                 activation:Callable=F.relu,
                 dropout:float=0.0):

        super(DuvenaudGCNClassifier, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # dense layer to do pre graph gather step
        # output layer, for pka prediction
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.activation = activation
        self.dense_layer = nn.Linear(n_hidden*2, n_hidden)
        self.classification_layer = nn.Linear(n_hidden,n_outputs)
        
    def forward(self, g):
        h = g.ndata['feat']

        h = DuvenaudGraphConvLayer(self.n_inputs, self.n_hidden, self.activation)(g,h)
        h = DuvenaudSoftmaxLayer(self.n_hidden, self.n_hidden)(g,h)
        f = h # save features after one layer for skip connections
        h = DuvenaudGraphConvLayer(self.n_hidden, self.n_hidden, self.activation)(g,h)
        h = DuvenaudSoftmaxLayer(self.n_hidden, self.n_hidden)(g,h)
        new_f = f + h

        g.ndata['f'] = f
        g.ndata['new_f'] = new_f
        fg = dgl.sum_nodes(g, 'f')
        new_fg = dgl.sum_nodes(g, 'new_f')

        # skip connections
        x = self.dense_layer(torch.cat((fg, new_fg), dim=1))
        x = F.relu(x)

        return self.classification_layer(x)

class SimpleGCNClassifier(nn.Module):
    def __init__(self, 
                 n_inputs:int,
                 n_hidden:int=64, 
                 n_outputs:int=1,
                 n_hidden_layers:int=2, 
                 activation:Callable=F.relu):

        super(SimpleGCNClassifier, self).__init__()

        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(
            GraphConvLayer(n_inputs, n_hidden, activation=activation)
        ) 
        # hidden layers
        for _ in range(n_hidden_layers-1):
            self.layers.append(
                GraphConvLayer(n_hidden, n_hidden, activation=activation)
            )

        self.classification_layer = nn.Linear(n_hidden, n_outputs)
        
    def forward(self, g):
        h = g.ndata['feat'] 
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h') 
        return self.classification_layer(hg)


