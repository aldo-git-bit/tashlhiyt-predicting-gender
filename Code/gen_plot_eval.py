"""
This piece of code is used to generated the grouped bar plots denoting F1 scores of the
different models on the different datasets. The plots generated using this are:
'Plots/plot_cv_nonhp_rocauc.png', 'Plots/plot_cv_nonhp_f1.png', 'Plots/plot_cv_nonhp_brier.png',
'Plots/plot_cv_hp_rocauc.png'
"""
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('Results/eval3_cv_nonhp.csv', index_col=0)

results_gr = pd.pivot_table(results, values='f1', index='Features', columns='Model', sort=False)
ax = results_gr.plot(kind='bar', rot=0, colormap='tab20b')
fig = ax.get_figure()
fig.set_size_inches(12, 8)
ax.grid(visible=True, axis='y', alpha=0.6, linestyle='dashed', color='black')
ax.set_title("F1 for different default models trained with different features for Multi-Classification")
ax.set_xlabel("Sets of Features used for Prediction")
ax.set_ylabel("F1 Score")
ax.legend(ncol=len(results_gr.columns))
ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.savefig("Plots/plot3_cv_nonhp_f1.png")