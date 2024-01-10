"""
We use simple models using features from different sets: "Phonological" , "Morphological"
and "Semantic" and their combinations to predict the word gender. We use five different
models with default setting, and perform a 5 fold cross validation study repeated 20 times.  
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

    def cross_val_study(self, model, repeats):
        scorer = make_scorer(roc_auc_score, needs_proba=True)
        scorer1 = make_scorer(brier_score_loss, needs_proba=True)
        kf = RepeatedKFold(n_repeats=repeats)
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc-auc', 'brier']
        scoring_dict = {
            'roc-auc':scorer, 'f1':make_scorer(f1_score), 
            'accuracy':make_scorer(accuracy_score), 'recall':make_scorer(recall_score),
            'precision':make_scorer(precision_score), 'brier': scorer1
        }
        cv_score = cross_validate(estimator=model, X=self.features, y=self.target,
                                scoring=scoring_dict, cv=kf)
        metric_scores = np.array([np.mean(cv_score['test_'+metric]) for metric in metrics])
        return np.round(metric_scores, 2)

def main():
    data_ = pd.read_csv('Data/final_data.csv', index_col=0)
    target = data_['Gender2']
    rfc = RandomForestClassifier(n_estimators=500, min_samples_split=10, min_samples_leaf=4,
                                max_features='sqrt', max_depth=90, bootstrap=False)
    abc = AdaBoostClassifier(n_estimators=200, learning_rate=1.0)
    gbd = GradientBoostingClassifier(n_estimators=100, min_samples_split=40, min_samples_leaf=5,
                                    max_depth=5, learning_rate=0.1)
    evc = VotingClassifier([('rfc', rfc), ('abc', abc), ('gbd', gbd)], voting='soft')
    svc = svm.SVC(probability=True)
    evaluations = []
    phonology = [2, 3, 4, 5, 6, 7]
    morphology = [8, 9, 10, 11, 12]
    semantic = [13, 14, 15]
    features_list = [(morphology, 'morphology'), (phonology, 'phonology'), (semantic, 'semantic'),
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
                                        'recall', 'f1', 'roc-auc', 'brier'])

    evaluations.to_csv('Results/eval_cv_hp.csv')

if(__name__=='__main__'):
    main()