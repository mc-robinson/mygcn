'''
Create atom and bond feature vectors
as in the code accompanying the original Duvenaud paper:
    - https://github.com/HIPS/neural-fingerprint/blob/2e8ef09/neuralfingerprint/features.py
    - https://github.com/HIPS/neural-fingerprint/blob/2e8ef09/neuralfingerprint/util.py

The code also relies on the code written in the dgl tutorials and 
by the chemoinformatics blogger iwatobipen:
    - https://docs.dgl.ai/tutorials/basics/4_batch.html
    - https://iwatobipen.wordpress.com/2019/02/01/try-gcn-qspr-with-pytorch-based-graph-library-rdkit-pytorch-dgl/

However, a few changes are made:
    - Bond featurization is slightly different to facilitate less need for 
      custom functions to deal with graphs. In our version, bond features are 
      aggregated based on the destination node. The aggregated features are then 
      essentially node features. 
    - The formal charge is included as an atom feature.

Additionally, this code provides the `build_mol_graph_from_smiles` function to 
build a dgl graph from a single smiles string. This graph can then be used as 
an input to a graph convolutional network (GCN) built in dgl. 

Much of this process is automated by the `get_graph_data` function, which is 
similar to the featurizer provided by Deepchem. It takes a csv input file and 
returns the necessary pytorch dataloaders for training and testing. 
'''

from rdkit import Chem
import numpy as np
import pandas as pd
import dgl
import torch
from torch.utils.data import DataLoader

# imports for typing
from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, \
                      Iterable, List, Mapping, NewType, Optional, Sequence, \
                      Tuple, TypeVar, Union                   
from types import SimpleNamespace

def one_of_k(value, allowed_values:Iterable) -> List[bool]:
    "Translates value into one hot encoded representation"
    
    # if not in list, set to last value, as per convention 
    # means last value of allowed_values should be set accordingly
    if value not in allowed_values:
        value = allowed_values[-1]
        
    # translate to true and false values
    return list(map(lambda x: value == x, allowed_values))

def get_atom_features(atom:Chem.rdchem.Atom) -> np.ndarray:
    "Concats all atom features together and returns numpy array of bools"
    
    # elements from Duvenaud's original code. Note the absense of H(optional)
    potential_atoms_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br',
                            'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
                            'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                            'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
                            'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    
    encoded_element_list = one_of_k(atom.GetSymbol(), potential_atoms_list)
    encoded_degree_list = one_of_k(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) 
    encoded_num_hs_list = one_of_k(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    encoded_fc_list = one_of_k(atom.GetFormalCharge(), [-1,-2,1,2,0])
    encoded_implicit_valence_list = one_of_k(atom.GetImplicitValence(),
                                             [0, 1, 2, 3, 4, 5])
    
    feature_vector = np.array(encoded_element_list + encoded_degree_list + 
                      encoded_num_hs_list + encoded_implicit_valence_list +
                      encoded_fc_list + [atom.GetIsAromatic()])
    
    return feature_vector

def get_bond_features(bond:Chem.rdchem.Bond) -> np.ndarray:
    "Concats all bond features together and returns numpy array of bools"
    
    potential_bond_types_list = [Chem.rdchem.BondType.SINGLE,
                                 Chem.rdchem.BondType.DOUBLE,
                                 Chem.rdchem.BondType.TRIPLE,
                                 Chem.rdchem.BondType.AROMATIC]
    
    encoded_bond_type_list = one_of_k(bond.GetBondType(),
                                      potential_bond_types_list)
    
    feature_vector = np.array(encoded_bond_type_list + 
                              [bond.GetIsConjugated()] +
                              [bond.IsInRing()])
    
    return feature_vector

def get_num_atom_features() -> int:
    "Uses example mol and `get_atom_features` to get length of feature vector"
    sample_mol = Chem.MolFromSmiles('CC')
    first_atom = sample_mol.GetAtoms()[0]
    return len(get_atom_features(first_atom))

def get_num_bond_features() -> int:
    "Uses example mol and `get_atom_features` to get length of feature vector"
    sample_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(sample_mol)
    first_bond = sample_mol.GetBonds()[0]
    return len(get_bond_features(first_bond))

def build_mol_graph_from_smiles(smiles_str:str, edge_features:bool=False,
                                   self_edges:bool=False) -> dgl.graph.DGLGraph:
    "Convert smiles string to dgl graph with relevant atom features."
    mol = Chem.MolFromSmiles(smiles_str)
    if not mol:
        raise ValueError("SMILES string not valid: ", smiles_str)
    
    mol_graph = dgl.DGLGraph()
    
    # add nodes into the graph, labels start at 0, which nicely matches RDKit
    num_atoms = len(mol.GetAtoms())
    mol_graph.add_nodes(num_atoms)

    node_features_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        node_features_list.append(get_atom_features(atom))
    
    
    # create edge list, a list of tuples
    edge_list = []
    edge_features_dict = {} # to hold the features for later
    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtom().GetIdx()
        end_atom_idx = bond.GetEndAtom().GetIdx()

        edge_list.append((begin_atom_idx, end_atom_idx))
        edge_list.append((end_atom_idx, begin_atom_idx)) #reverse

        edge_features_dict[(begin_atom_idx, end_atom_idx)] = \
        get_bond_features(bond)
        edge_features_dict[(end_atom_idx, begin_atom_idx)] = \
        get_bond_features(bond) # reverse
        
    src, dst = tuple(zip(*edge_list))
    mol_graph.add_edges(src, dst)
    # mol_graph.add_edges(dst, src) already added edges in both dirs
    # since we included both edges in edge list

    if edge_features:
        # add edge features (need to do for both directions)
        # we sum the feature vectors of all nodes coming into a given node
        # the sum vector is then added to the node feature vector

        # go through and find dstination
        num_bond_features = 6
        incoming_edges_dict = dict.fromkeys(range(num_atoms),
                                            np.zeros(num_bond_features))
        for edge in edge_list:
            dst_node = edge[1]
            old_entry = incoming_edges_dict[dst_node]
            new_entry = old_entry + (edge_features_dict[edge].astype(int))
            incoming_edges_dict[dst_node] = new_entry
            
        node_features_from_bonds = torch.Tensor([incoming_edges_dict[x]
                                                 for x in incoming_edges_dict])
        node_features_from_nodes = torch.Tensor(node_features_list)
        mol_graph.ndata['feat'] = torch.cat(
                                    (node_features_from_nodes,
                                     node_features_from_bonds),
                                    dim=1
                                  )
    
    # note that self_edges will have no features, but make it easy to
    # message pass to the same node, as is required in Duvenaud and Deepchem
    if self_edges:
        self_edges_list = []
        for i in range(num_atoms):
            self_edges_list.append((i,i))
        src, dst = tuple(zip(*self_edges_list))
        mol_graph.add_edges(src, dst)
    
    return mol_graph

def get_graph_data(csv_file_path:str, smiles_field:str, labels_field:str,
                   normalize_labels:bool=False, train_size:float=0.8, 
                   valid_size:float=0.1, seed:int=42, self_edges:bool=True,
                   edge_features:bool=False):
    "Construct datasets from csv file with smiles codes and labels"
    
    df = pd.read_csv(csv_file_path)
    
    # do train/valid/test split based on given percentages
    train_df, valid_df, test_df = np.split(
        df.sample(frac=1,random_state=seed),
        [
            int(train_size*len(df)),
            int((train_size+valid_size)*len(df))
        ]
    )
    
    train_smiles_strs = list(train_df[smiles_field])
    train_labels = list(train_df[labels_field])
    train_graphs = [
        build_mol_graph_from_smiles(smiles, self_edges=self_edges, 
                                    edge_features=edge_features)
        for smiles in train_smiles_strs
    ]
    
    valid_smiles_strs = list(valid_df[smiles_field])
    valid_labels = list(valid_df[labels_field])
    valid_graphs = [
        build_mol_graph_from_smiles(smiles, self_edges=self_edges, 
                                    edge_features=edge_features)
        for smiles in valid_smiles_strs
    ]
    
    test_smiles_strs = list(test_df[smiles_field])
    test_labels = list(test_df[labels_field])
    test_graphs = [
        build_mol_graph_from_smiles(smiles, self_edges=self_edges,
                                    edge_features=edge_features)
        for smiles in test_smiles_strs
    ]

    if normalize_labels:
        # note that we can only use train mean and std
        # or else snooping on valid and test data
        train_mean = np.mean(train_labels)
        train_std = np.std(train_labels)

        train_labels = list((np.array(train_labels)-train_mean)/train_std)
        valid_labels = list((np.array(valid_labels)-train_mean)/train_std)
        test_labels = list((np.array(test_labels)-train_mean)/train_std)
    
    # define function to be used in building Dataloaders
    # Notice that we batch the graphs together, this speeds things up
    def collate(samples:List[tuple]):
        # The input `samples` is a list of pairs
        #  (graph, label).
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)
    
    train_set = list(zip(train_graphs, train_labels))
    valid_set = list(zip(valid_graphs, train_labels))
    test_set = list(zip(test_graphs, test_labels))
    
    train_dl = DataLoader(train_set, batch_size=32, collate_fn=collate,
                          shuffle=True)
    valid_dl = DataLoader(valid_set, batch_size=32, collate_fn=collate)
    test_dl = DataLoader(test_set, batch_size=32, collate_fn=collate)
    
    return train_dl, valid_dl, test_dl
    


