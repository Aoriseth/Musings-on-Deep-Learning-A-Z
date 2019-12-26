import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# %%

label_encoder = LabelEncoder()
X[:, 1] = label_encoder.fit_transform(X[:, 1])
X[:, 2] = label_encoder.fit_transform(X[:, 2])
transformer = ColumnTransformer(transformers=[("Country", OneHotEncoder(categories='auto'), [1])],
                                remainder='passthrough')
X = transformer.fit_transform(X)
X = X[:, 1:]

# %%
# Extract training and testing sets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#%%
# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
print(X)
print(Y)

# %%
# plt.scatter([1, 2, 3], [5, 5, 8])
# plt.show()
