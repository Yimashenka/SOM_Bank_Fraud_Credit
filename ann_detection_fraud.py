# CREATING THE ANN TO CREATE A AUTO FRAUD DETECTION MODEL
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# IMPORTING LIBRAIRIES AND DEPENDENCIES
from visualizing_results_som import fids
from preprocessing_data import dataset

customers = dataset.iloc[:, 1:].values

# CREATE THE DEPENDANT VARIABLE
is_fraud = np.zeros((len(dataset)))

for i in range (len(dataset)):
    if dataset.iloc[i, 0] in fids:
        is_fraud[i] = 1

# FEATURE SCALING
sc = StandardScaler()
customers = sc.fit_transform(customers)

# BUILDING THE ANN
ann = Sequential()

ann.add(
    Dense(
        units=2,
        kernel_initializer='uniform',
        activation='relu',
        input_dim=15
    )
)

ann.add(
    Dense(
        units=1,
        kernel_initializer='uniform',
        activation='sigmoid'
    )
)

ann.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

ann.fit(
    customers,
    is_fraud,
    batch_size=1,
    epochs=5
)

