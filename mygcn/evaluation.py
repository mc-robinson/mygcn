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

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

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

def evaluate_regressor(model:nn.Module, test_dl:DataLoader,
                        loss_func:Callable) -> None:
    "evaluate a pytorch graph model for regression"

    y_pred = []
    y_true = []
    test_loss = 0
    for bg, labels in test_dl:
        model.eval()

        bg.set_e_initializer(dgl.init.zero_initializer)
        bg.set_n_initializer(dgl.init.zero_initializer)

        if model(bg).shape[1] == 1:
            labels = labels.reshape(-1,1)

        loss = loss_func(model(bg), labels)
        test_loss += loss.detach().item()

        y_pred += list(model(bg).detach().numpy().reshape(-1,))
        y_true += list(np.array(labels.reshape(-1,)))
        
    print('test_loss: ',test_loss/len(test_dl.dataset))
    print('***')

    print('RMSE: ', np.sqrt(mean_squared_error(y_true, y_pred)))
    print('RMSE CI: ', RMSE_CI(y_true, y_pred))
    print('***')

    print('MAE: ', mean_absolute_error(y_true, y_pred))
    print('MAE CI: ', MAE_CI(y_true, y_pred))
    print('***')

    print('R^2: ', r2_score(y_true, y_pred))
    print('R^2 CI: ', R2_CI(y_true, y_pred))

def evaluate_dc_classifier(model, testset,
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

def R2_CI(y_true:List, y_pred:List):
    "code using the Fischer transform to compute R2 CI"

    r2 = r2_score(y_true, y_pred)
    r = np.sqrt(r2)
    N = len(y_true)

    def pearson_confidence(r, num, interval=0.95):
        from scipy.stats import pearsonr
        from scipy.stats import norm
        import math
        """
        FROM Pat Walters!!! (https://github.com/PatWalters/metk)
        Calculate upper and lower 95% CI for a Pearson r (not R**2)
        Inspired by https://stats.stackexchange.com/questions/18887
        :param r: Pearson's R
        :param num: number of data points
        :param interval: confidence interval (0-1.0)
        :return: lower bound, upper bound
        """
        stderr = 1.0 / math.sqrt(num - 3)
        z_score = norm.ppf(interval)
        delta = z_score * stderr
        lower = math.tanh(math.atanh(r) - delta)
        upper = math.tanh(math.atanh(r) + delta)
        return lower, upper

    lower_CI, upper_CI = [x**2 for x in pearson_confidence(r,N)]
    return lower_CI, upper_CI

def RMSE_CI(y_true:List, y_pred:List):
    "Computes the RMSE CI using the formula from Nicholls (2014)"
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    N = len(y_pred)

    s2 = rmse**2
    lower_limit = s2 - 1.96*s2*np.sqrt(2/(N-1))
    upper_limit = s2 + 1.96*s2*np.sqrt(2/(N-1))
    return np.sqrt(lower_limit), np.sqrt(upper_limit)

def MAE_CI(y_true:List, y_pred:List, n_boostraps:int=1000):
    "code to bootstrap the MAE score: adapted from Ogrisel's AUC code"

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        score = mean_absolute_error(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return [confidence_lower, confidence_upper]

