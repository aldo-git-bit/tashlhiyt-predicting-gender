"""
We use simple models using features from different sets: "Phonological" , "Morphological"
and "Semantic" and their combinations to predict the word gender. This is a multi-classification
problem as compared to the previous one which was only binary. We use five different models with
default setting, and perform a 5 fold cross validation study repeated 20 times.  
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import *

class model_evaluate:
    def __init__(self, features, target) -> None:
        self.features = features
        self.target = target

    def cross_val_study(self, model):
        scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')
        kf = RepeatedKFold(n_repeats=20)
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc-auc']
        scoring_dict = {
            'roc-auc':scorer, 'f1':make_scorer(f1_score, average='macro'), 
            'accuracy':make_scorer(accuracy_score),
            'recall':make_scorer(recall_score, average='macro'),
            'precision':make_scorer(precision_score, average='macro')
        }
        cv_score = cross_validate(estimator=model, X=self.features, y=self.target,
                                scoring=scoring_dict, cv=kf)
        metric_scores = np.array([np.mean(cv_score['test_'+metric]) for metric in metrics])
        return np.round(metric_scores, 2)


data_ = pd.read_csv('Data/final_data.csv', index_col=0)
target = data_['Gender3']
rfc = RandomForestClassifier()
abc = AdaBoostClassifier()
gbd = GradientBoostingClassifier()
evc = VotingClassifier([('rfc', rfc), ('abc', abc), ('gbd', gbd)], voting='soft')
svc = svm.SVC(probability=True)
evaluations = []
phonology = [2, 3, 4, 5, 6, 7]
morphology = [8, 9, 10, 11, 12]
semantic = [13, 14, 15]
features_list = [(phonology, 'phonology'), (morphology, 'morphology'), (semantic, 'semantic'),
                 (phonology+morphology, 'phon+morph'), (phonology+semantic, 'phon+sem'),
                 (morphology+semantic, 'morph+sem'), (phonology+semantic+morphology, 'all')]
models = [(rfc, 'rfc'), (abc, 'abc'), (gbd, 'gbd'), (svc, 'svc'), (evc, 'evc')]
for feat in features_list:
    features = np.array(data_[data_.columns[feat[0]]])
    features_full = model_evaluate(features, target)
    for model in models:
        res = list(features_full.cross_val_study(model[0]))
        res = [feat[1], model[1]] + res
        evaluations.append(res)

evaluations = pd.DataFrame(evaluations,
                           columns=['Features', 'Model', 'accuracy', 'precision',
                                     'recall', 'f1', 'roc-auc'])

evaluations.to_csv('Results/eval3_cv_nonhp.csv')