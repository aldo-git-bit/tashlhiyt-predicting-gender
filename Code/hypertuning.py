"""
The three models: RandomForestClassifier, AdaboostClassifier and GradientBoostClassifier
are hypertuned here. 
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

def hypertune_rfc(train_feat, train_targ):
    #Hypertuning for Random Forest Classifier
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    rfc = RandomForestClassifier()
    scorer = make_scorer(roc_auc_score, needs_proba=True)
    rfc_search = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, scoring=scorer,
                                    n_iter = 50, cv = 5, random_state=1)
    rfc_search.fit(train_feat, train_targ)
    print(rfc_search.best_params_, rfc_search.best_score_)


def hypertune_adb(train_feat, train_targ):
    #Hypertuning for Adaboost Classifier
    learning_rate = [0.1, 0.5, 1.0, 2.0]
    n_estimators = [10, 20, 30, 50, 60, 80, 100, 200]
    search_grid = {'learning_rate':learning_rate, 'n_estimators':n_estimators}
    scorer = make_scorer(roc_auc_score, needs_proba=True)
    abc_search = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=search_grid, cv=5, scoring=scorer) 
    abc_search.fit(train_feat, train_targ)
    print(abc_search.best_params_, abc_search.best_score_)

def hypertune_gbd(train_feat, train_targ):
    #Hypertuning for Gradient Boosting Classifier
    max_depth = [3, 5, 10, 20, 50]
    min_samples_leaf = [1, 5, 10, 20]
    min_samples_split = [2, 10, 20, 40]
    learning_rate = [0.1, 0.2, 0.5, 1.0]
    n_estimators = [10, 20, 40, 50, 100, 200, 500]
    gbd = GradientBoostingClassifier()
    search_grid = {'max_depth':max_depth, 'min_samples_leaf':min_samples_leaf, 'min_samples_split':min_samples_split,
                'learning_rate':learning_rate, 'n_estimators':n_estimators}
    scorer = make_scorer(roc_auc_score, needs_proba=True)
    gbd_search = RandomizedSearchCV(gbd, param_distributions=search_grid, cv=5, scoring=scorer, n_iter=250)
    gbd_search.fit(train_feat, train_targ)
    print(gbd_search.best_params_, gbd_search.best_score_)



data_ = pd.read_csv('Data/final_data.csv', index_col=0)
features = np.array(data_[data_.columns[2:]])
target = data_['Gender2']
train_feat, test_feat, train_targ, test_targ = train_test_split(features, target,
                                                                test_size=0.20, random_state=1)
hypertune_rfc(train_feat, train_targ)
hypertune_adb(train_feat, train_targ)
hypertune_gbd(train_feat, train_targ)