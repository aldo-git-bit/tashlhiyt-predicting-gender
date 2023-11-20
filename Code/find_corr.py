"""
We encode the features and visualize the correlation among features and between
features and targets. From the plot it is evident that sexgen is highly correlated
with Human and Animate, so we remove sexgen from further considerations.

Input - int_data.csv
Output - correlation.png and final_data.csv
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Data/int_data.csv', index_col=0)

"""
Encoding all the categorical columns in the data is in the following way:
Any Column C with N categories is labelled with 0, 1, 2, 3, .., N-1. 
"""

le = LabelEncoder()
categorical_columns = [0, 1] + [i for i in range(6, len(data.columns))]
for i in categorical_columns:
    data[data.columns[i]] = le.fit_transform(data[data.columns[i]])

"""
Plotting correlation between columns of the features. The image is saved under
plots folder as correlation.png. The Pearson correlation coefficient is calculated.
"""

plt.rcParams['figure.figsize'] = (12, 10)
corr_mat = data.corr()
sns.heatmap(corr_mat, cmap='Set1')
plt.title("Pearson Correlation between Pairs of Features")
plt.savefig('Plots/correlation.png')

"""
From the Correlation plot, it is evident that human and animate are highly correlated
with sexgen. So we remove sexgen column from the data
"""
data.drop(columns=['sexgen'], inplace=True)
data.to_csv('Data/final_data.csv')