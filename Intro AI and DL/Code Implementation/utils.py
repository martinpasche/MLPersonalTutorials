import numpy as np
import torch
import matplotlib.pyplot as plt

# Plot the dataset
def plot_data(ax, X, Y, marker='o'):
    plt.axis('off')
    ax.scatter(X[:, 0], X[:, 1], s=1, c=Y, cmap='bone', marker=marker)

# plot the decision boundary of our classifier
def plot_decision_boundary(ax, X, Y, classifier, use_tensor=False):
    
    # Define the grid on which we will evaluate our classifier
    x_min, x_max = -1.5, 2.5
    y_min, y_max = -1, 1.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),
                         np.arange(y_min, y_max, .01))

    to_forward = np.array(list(zip(xx.ravel(), yy.ravel())))
    to_forward_tensor = torch.from_numpy(np.array(list(zip(xx.ravel(), yy.ravel())))).float()
    
    # forward pass on the grid, then convert to numpy for plotting
    if use_tensor:
        Z = classifier.forward(to_forward_tensor)
    else:
        Z = classifier.forward(to_forward)
    Z = Z.reshape(xx.shape)
    
    # plot contour lines of the values of our classifier on the grid
    ax.contourf(xx, yy, Z>0.5, cmap='Blues')
    
    # then plot the dataset
    plot_data(ax, X,Y)