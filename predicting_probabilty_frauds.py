# PREDICTING THE PROBABILITY OF FRAUDS (USING THE ANN)

# IMPORTING THE DEPENDENCIES
from ann_detection_fraud import ann, customers, np, dataset

# PREDICTING
y_pred = ann.predict(customers)

y_pred = y_pred*100     #saw the results as %
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)

# SORT THE IDS BY % OF FRAUDS
y_pred = y_pred[y_pred[:, 1].argsort()]     #[id, % of fraud]

np.savetxt('results/frauds.csv', y_pred, delimiter=",")
