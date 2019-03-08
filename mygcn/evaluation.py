'''
Code to evaluate pytorch classification/regression models given a dataloader

Note that the code for computing uncertainty of auc scores is taken from:
    - https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals
    - https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
'''

import numpy as np
import pandas as pd
import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# get test statistics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.preprocessing import label_binarize

# imports for typing
from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, \
                      Iterable, List, Mapping, NewType, Optional, Sequence, \
                      Tuple, TypeVar, Union                   
from types import SimpleNamespace

def evaluate_classifier(model:nn.Module, test_dl:DataLoader,
                        loss_func:Callable, classes:List[int]=[0,1]) -> None:
    "evaluate a pytorch graph model for classification"

    y_pred = []
    y_true = []
    y_prob = []
    prob_arr = []

    test_loss = 0

    for bg, labels in test_dl:
        model.eval()
        
        bg.set_e_initializer(dgl.init.zero_initializer)
        bg.set_n_initializer(dgl.init.zero_initializer)
        
        logit = model(bg)
        probs = torch.softmax(logit,1).detach().numpy()
        prob_arr.append(probs)
        predictions = np.argmax(probs,1)


        y_pred += list(predictions)
        y_true += list(labels)
        y_prob += list(probs[:,1])
        
        loss = loss_func(logit, labels)
        test_loss += loss.detach().item()
        
    print('test_loss: ',test_loss/len(test_dl))
    print('accuracy: ', accuracy_score(y_true, y_pred))
    print('classification report: \n',
            classification_report(y_true, y_pred))

    if len(classes)==2:
        print('roc-auc: ', roc_auc_score(y_true, y_prob))
        print('bootstrapped roc-auc: ', bs_roc_auc_score(y_true, y_prob))

    else:
        y_test = label_binarize(y_true, classes=classes)
        n_classes = y_test.shape[1]

        prob_arr = np.concatenate([x for x in prob_arr], axis=0)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        bs_roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], prob_arr[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            bs_roc_auc[i] = bs_roc_auc_score(y_test[:, i], prob_arr[:, i])
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), prob_arr.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        bs_roc_auc['micro'] = bs_roc_auc_score(y_test.ravel(), prob_arr.ravel())

        print("micro auc score and score for each class: ")
        for key in roc_auc:
            print(key,' : ', roc_auc[key])
        print("bootstrapped micro auc score and score for each class: ")
        for key in bs_roc_auc:
            print(key,' : ', bs_roc_auc[key])

def evaluate_dc_calssifier(model, testset,
                           classes:List[int]=[0,1]) -> None:
    "evaluate a DeepChem `model` for classification on a dc `dataset`: testset"

    y_true = list(testset.y[:,0].astype(int))
    
    prob_arr = np.array([x[0] for x in model.predict(testset)])
    y_pred = list(np.argmax(prob_arr, axis=1))

    print('accuracy: ', accuracy_score(y_true, y_pred))
    print('classification report: \n',
            classification_report(y_true, y_pred))

    if len(classes) == 2:
        y_prob = list(prob_arr[:,-1])
        print('roc-auc: ', roc_auc_score(y_true, y_prob))
        print('bootstrapped roc-auc: ', bs_roc_auc_score(y_true, y_prob))

    else:
        y_test = label_binarize(y_true, classes=classes)
        n_classes = y_test.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        bs_roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], prob_arr[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            bs_roc_auc[i] = bs_roc_auc_score(y_test[:, i], prob_arr[:, i])
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), prob_arr.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        bs_roc_auc['micro'] = bs_roc_auc_score(y_test.ravel(), prob_arr.ravel())

        print("micro auc score and score for each class: ", roc_auc)
        print("bootstrapped micro auc score and score for each class: ", bs_roc_auc)

def bs_roc_auc_score(y_true:List, y_prob:List, n_boostraps:int=1000):
    "code to bootstrap the auc score: copied from Ogrisel's SO answer"

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_prob) - 1, len(y_prob))
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return [confidence_lower, confidence_upper]




