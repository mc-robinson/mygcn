'''
Code to run a simple graph conv net on the molecules
with a unique pka
'''

import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel

graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
loader = dc.data.data_loader.CSVLoader( tasks=['pka'],
                                        smiles_field="CANONICAL_SMILES",
                                        featurizer=graph_featurizer )

# NOTE: setting the seed for dc's RandomSplitter() does not work
# so best to do the splitting of datasets yourself
# dataset = loader.featurize( './unique_pkas.csv' )

train_dataset = loader.featurize( '../data/dw_acidic_unique_train.csv' )
valid_dataset = loader.featurize( '../data/dw_acidic_unique_valid.csv' )
test_dataset = loader.featurize( '../data/dw_acidic_unique_test.csv' )

# splitter = dc.splits.RandomSplitter()
# train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset, seed=42)

transformers = [
        dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]

for dataset in [train_dataset, valid_dataset, test_dataset]:
        for transformer in transformers: 
                    dataset = transformer.transform(dataset)

model = GraphConvModel(n_tasks=1, mode='regression',
                       tensorboard=True,  model_dir='models/',
                       dropout=0.5, graph_conv_layers=[64,64])

# Need to use the following hackish code to track the validation loss 
# while fitting with DeepChem, this is how I track overfitting.
valid_loss = 10000000
while valid_loss > 50:
        # Fit trained model 
        model.fit(train_dataset, nb_epoch= 1)
        # checkpoint_interval causes the model not to save a checkpoint. 
        valid_loss = model.fit(valid_dataset, checkpoint_interval=0)
        print("valid loss: ", valid_loss)
        # This will restore the model to the fit from the train dataset 
        model.restore()
        model.save()
