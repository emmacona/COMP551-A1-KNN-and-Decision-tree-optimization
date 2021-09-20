import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

# 1. Load the datasets into Pandas dataframes
names1=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'over-50K']
dataset1_train = pd.read_csv('adult_train.csv', index_col=False, names=names1, sep=',\s', na_values=['?'], engine='python')
dataset1_test = pd.read_csv('adult_test.csv', index_col=False, names=names1, sep=',\s', na_values=['?'], engine='python')

print(dataset1_train.shape)

# Clean dataset #1
# Remove instances with missing/invalid data entries
dataset1_train.dropna(axis=0, how='any', inplace=True)
print(dataset1_train.shape)

# Encode Over 50K as 1
dataset1_train['over-50K'] = dataset1_train['over-50K'].map({'>50K': 1, '<=50K': 0})

# Encode Male as 1
dataset1_train['sex'] = dataset1_train['sex'].map({'Male': 1, 'Female': 0})

print(dataset1_train['native-country'].describe()) # Since 91% of native-country data is US, we will categorize this column as "US" and "non-US".

# Encode US as 1 and non-US as 0
dataset1_train['native-country'] = np.where(dataset1_train['native-country']=='United-States', 1, 0)

# Plots
plt.hist(dataset1_train['race'], histtype='bar')
plt.hist(dataset1_train['hours-per-week'], histtype='bar')
plt.hist(dataset1_train['education'], histtype='bar')


# One-hot encoding
# Split into training, validation, and tests
X = dataset1_train.drop(['over-50K'], axis=1) # drop column about over or under-50K (salary)
Y = dataset1_train['over-50K']

X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.15, shuffle=True)
print(Y_train.shape, Y_test.shape)
