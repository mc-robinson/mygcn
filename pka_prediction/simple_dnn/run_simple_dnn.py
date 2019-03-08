'''
Code to run a simple feedforward net on the molecules
with a unique pka
'''

import deepchem as dc

fp_featurizer = dc.feat.CircularFingerprint(radius=1, size=2048)
loader = dc.data.data_loader.CSVLoader( tasks=['pka'],
                                        smiles_field="CANONICAL_SMILES",
                                        featurizer=fp_featurizer )

# Split datasets myself since seed in RandomSplitter() does not work
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

# model = GraphConvModel(n_tasks=1, mode='regression',
#                        tensorboard=True,  model_dir='models/',
#                        dropout=0.5, graph_conv_layers=[64,64])

model = dc.models.MultitaskRegressor(n_tasks=1,
                                     n_features= 2048,   
                                     layer_sizes=[256,256],
                                     model_dir='dnn_models/')
        
# Hackish code used for training because I want to track valid_loss
# to see if overfitting is occurring 
valid_loss = 10000000
while valid_loss > 50:
        # Fit trained model 
        model.fit(train_dataset, nb_epoch= 10)
        # checkpoint_interval causes the model not to save a checkpoint. 
        valid_loss = model.fit(valid_dataset, checkpoint_interval=0)
        print("valid loss: ", valid_loss)
        # This will restore the model to the fit from the train dataset 
        model.restore()
        model.save()
