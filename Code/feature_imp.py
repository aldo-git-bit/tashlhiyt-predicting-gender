"""
The Relative feature importance for the different models trained on different 
pools of features are computed and stored in 'Results/rel_importance_nonhp.csv'
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.metrics import *
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from time import time

def feat_importance(model, features, target):
    concat_results = []
    start = time()
    for state in range(1, 20):
        train_feat, test_feat, train_targ, test_targ = train_test_split(features, target,
                                                                         test_size=0.20, random_state=state)
        model.fit(train_feat, train_targ)
        scorer = make_scorer(roc_auc_score, needs_proba=True)
        results = permutation_importance(model, test_feat, test_targ, n_repeats=20,
                                        random_state=45, n_jobs=2, scoring=scorer)
        concat_results.append(results.importances_mean)
    print(time()-start)
    return np.mean(np.array(concat_results), axis=0)

data_ = pd.read_csv('Data/final_data.csv', index_col=0)
target = data_['Gender2']
phonology = [2, 3, 4, 5, 6, 7]
morphology = [8, 9, 10, 11, 12]
semantic = [13, 14, 15]
features_list = [(phonology, 'phonology'), (morphology, 'morphology'), (semantic, 'semantic'),
                 (phonology+morphology, 'phon+morph'), (phonology+semantic, 'phon+sem'),
                 (morphology+semantic, 'morph+sem'), (phonology+semantic+morphology, 'all')]
features = np.array(data_[data_.columns[features_list[0][0]]])
rfc = RandomForestClassifier()
abc = AdaBoostClassifier()
gbd = GradientBoostingClassifier()
evc = VotingClassifier([('rfc', rfc), ('abc', abc), ('gbd', gbd)], voting='soft')
svc = svm.SVC(probability=True)
models = [(rfc, 'rfc'), (abc, 'abc'), (gbd, 'gbd'), (svc, 'svc'), (evc, 'evc')]
evaluations = []
for model in models:
    for feat in features_list:
        cols_ = data_.columns[feat[0]]
        features = np.array(data_[cols_])
        imp_ = feat_importance(model[0], features, target)
        for i in range(len(imp_)):
            evaluations.append([model[1], feat[1], cols_[i], imp_[i]])

evaluations = pd.DataFrame(evaluations, columns=['Model', 'Features', 'Feature', 'ROC-AUC decrease'])
evaluations.to_csv('Results/rel_importance_nonhp.csv')