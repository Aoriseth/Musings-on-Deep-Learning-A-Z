import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 4].values

# %%
print(X)
print(Y)

# %%
plt.scatter([1, 2, 3], [5, 5, 8])
plt.show()
