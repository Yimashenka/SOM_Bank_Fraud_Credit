# IMPORTING THE LIBRAIRIES
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# IMPORTING THE DATASET
dataset = pd.read_csv('data/Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values  # contains if yes or no the application was
                                # approved

# FEATURE SCALING
sc = MinMaxScaler(
    feature_range=(0,1)
)
X = sc.fit_transform(X)
