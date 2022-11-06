# IMPORTING THE DEPENDENCIES
from minisom import MiniSom
from preprocessing_data import X


# TRAINING THE SOM
# We will build a 10x10 grid, no need for much as we don't have a lots of data
# (x and y parameters in MiniSom class). Our data has 16 row, minus one (the
# label). We don't need the user ID normally, but here, as we are looking for
# fraude, we are interesting by keeping the customer ID to retrieve the
# frauds.
# Sigma corresponding at the circle radisu, and the leanring rate, the more
# close to one, the more the convergence will be fast (but not particuraly
# acurate).

# The larger is the MID, the closer to white the color will be.

som = MiniSom(
    x=10,
    y=10,
    input_len=15,
    sigma=1.0,
    learning_rate=0.5)

som.random_weights_init(X)
som.train_random(
    data=X,
    num_iteration=100
)
