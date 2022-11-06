# VISUALIZING THE RESULTS
# We are not building a classical graph. We are creating a Self-Organizing Map,
# so we will not use matplotlib right now.

# IMPORTING THE LIBRAIRIES
from matplotlib.pylab import bone, pcolor, colorbar, plot, show
from som_training import som
from preprocessing_data import X, Y


# First we need to initialize the figure, that is the window that will contain
# the map => bone()

# Next, we have to put the different winning nodes on the map. Different co-
# -lors corresponding to the different range values of the Mean Interneuron
# Distance = pcolor().

bone()
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r', 'g']

#for each customers we're going to get the winning node, and dependaning on
#wheter the customers get the approval or not, we're going to color this winning
# node by red circle if not, green square if yes.
# x corresponding to values vector of custommer i in our dataset
for i, x in enumerate(X):
  w = som.winner(x) #get the winning node of the customer x
  plot(w[0] + 0.5,
       w[1] + 0.5,
       markers[Y[i]],
       markeredgecolor=colors[Y[i]],
       markerfacecolor='None',
       markersize=10,
       markeredgewidth=2
       )
show()