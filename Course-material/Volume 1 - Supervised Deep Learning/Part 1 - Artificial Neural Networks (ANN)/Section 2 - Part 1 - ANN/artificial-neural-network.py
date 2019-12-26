import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# %%
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

label_encoder = LabelEncoder()
X[:, 1] = label_encoder.fit_transform(X[:, 1])
X[:, 2] = label_encoder.fit_transform(X[:, 2])
transformer = ColumnTransformer(transformers=[("Country", OneHotEncoder(categories='auto'), [1])],
                                remainder='passthrough')
X = transformer.fit_transform(X)
X = X[:, 1:]

# %%
print(X)
print(Y)

# %%
# plt.scatter([1, 2, 3], [5, 5, 8])
# plt.show()
