import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
models = ['rfc', 'abc', 'gbd', 'svc', 'evc']
for model in models:
    rel_importances = pd.read_csv('Results/rel_importance_nonhp.csv', index_col=0)
    model_spec = rel_importances[rel_importances['Model']==model]
    features_ = ['phonology', 'morphology', 'semantic']
    features1_ = ['phon+morph', 'phon+sem', 'morph+sem']
    sub_selection = model_spec[model_spec['Features'].isin(features_)]
    sub_selection = pd.pivot_table(sub_selection, values='ROC-AUC decrease',
                                    index='Features', columns='Feature', sort=False)
    sub_selection1 = model_spec[model_spec['Features'].isin(features1_)]
    sub_selection1 = pd.pivot_table(sub_selection1, values='ROC-AUC decrease',
                                    index='Features', columns='Feature', sort=False)
    sub_selection2 = model_spec[model_spec['Features']=='all']
    sub_selection2 = pd.pivot_table(sub_selection2, values='ROC-AUC decrease',
                                    index='Features', columns='Feature', sort=False)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1 = fig.add_subplot(2, 2, 3)
    sub_selection.plot(ax=ax, kind='bar', rot=0, colormap='tab20b')
    ax.get_legend().remove()
    ax.set_xlabel(xlabel='')
    sub_selection1.plot(ax=ax1, kind='bar', rot=0, colormap='tab20b')
    ax1.get_legend().remove()
    ax1.set_xlabel(xlabel='')
    sub_selection2.plot(ax=ax2, kind='bar', rot=0, colormap='tab20b')
    ax2.set_xlabel(xlabel='')
    #ax.legend(ncol=len(sub_selection.columns))
    fig.suptitle(f'Relative Importance for features when {model} is trained using a specified features pool')
    fig.savefig('Plots/plot_rel_imp'+model+'.png')