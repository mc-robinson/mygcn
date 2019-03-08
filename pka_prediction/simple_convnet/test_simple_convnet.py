'''
Code to test a simple graph conv net on the molecules
with a unique pka
'''

import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel

graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
loader = dc.data.data_loader.CSVLoader( tasks=['pka'],
                                         smiles_field="CANONICAL_SMILES",
                                        featurizer=graph_featurizer )

# Split dataset myself, since DeepChem seed in splitter is not working
# splitter = dc.splits.RandomSplitter()
# train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset, seed=42)

train_dataset = loader.featurize( '../data/dw_acidic_unique_train.csv' )
valid_dataset = loader.featurize( '../data/dw_acidic_unique_valid.csv' )
test_dataset = loader.featurize( '../data/dw_acidic_unique_test.csv' )

transformers = [
        dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]

for dataset in [train_dataset, valid_dataset, test_dataset]:
        for transformer in transformers:
                    dataset = transformer.transform(dataset)

# print('shape of the dataset')
# print(train_dataset.X.shape)
# print(train_dataset.X[0].get_atom_features())
# print(train_dataset.X[0].get_atom_features().shape)

model = GraphConvModel.load_from_dir('models')
model.restore()

train_scores = model.evaluate(
                train_dataset,
                [dc.metrics.Metric(dc.metrics.rms_score),
                 dc.metrics.Metric(dc.metrics.r2_score),
                 dc.metrics.Metric(dc.metrics.mae_score)]
                )
print('train scores')
print(train_scores)

valid_scores = model.evaluate(
                valid_dataset,
                [dc.metrics.Metric(dc.metrics.rms_score),
                 dc.metrics.Metric(dc.metrics.r2_score),
                 dc.metrics.Metric(dc.metrics.mae_score)]
                )
print('valid scores')
print(valid_scores)


test_scores = model.evaluate(
                test_dataset,
                [dc.metrics.Metric(dc.metrics.rms_score),
                 dc.metrics.Metric(dc.metrics.r2_score),
                 dc.metrics.Metric(dc.metrics.mae_score)]
                )
print('test_scores')
print(test_scores)

import matplotlib.pyplot as plt
import numpy as np

train_predictions = model.predict(train_dataset)
test_predictions = model.predict(test_dataset)

plt.figure(figsize=(8,6))

plt.scatter(train_dataset.y, train_predictions, alpha=0.3, label='train')
plt.scatter(test_dataset.y, test_predictions, alpha=0.3, label='test')
plt.xlim(-10,20)
plt.ylim(-10,20)
x_vals = np.linspace(-10,20,100)
plt.plot(x_vals,x_vals,'r')

plt.legend()

plt.xlabel("Experimental pKa")
plt.ylabel("GCN Prediction")
plt.savefig('gcn_train_test_plot.png')

