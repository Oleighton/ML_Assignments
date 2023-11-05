import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score

# this assignment is focused on determning the best kernel for SVMs and evaluting the 
# computational tradeoffs of SVMs vs MLPs models

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

def define_decision_boundary_levels(X, clf):
     # Define the levels for the decision boundary
    # source: stack over flow https://stackoverflow.com/questions/51297423/plot-scikit-learn-sklearn-svm-decision-boundary-surface
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                         np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return xx, yy, Z

def plot_decision_boundary(X, y, clf, xx, yy, Z, kernel):
    # In summary, these lines of code plot 
    # the data points, create a grid for the 
    # entire feature space, calculate the decision
    # function values for each point in the grid, 
    # and then plot the decision boundary as a contour line.
    # The decision boundary is the line where the decision function equals 0, separating the two classes.
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='b', marker='o', label='Class A')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', marker='s', label='Class B')
    plt.contour(xx, yy, Z, levels=[0], colors='k')
    plt.xlabel('X Axis Value')
    plt.ylabel('Y Axis Value')
    plt.title(f'Kernel Type: {kernel}')
    plt.legend(loc='best')
    plt.show()
    
def find_best_kernel_circles(kernel):
    # make a syntehtic data set containing 1000 samples with the inner circle size being
    # 0.3 times the outer circle. The noise parameter adds random noise to the data points
    # random state is used to set the random seed for data generation to ensure reproducibliity
    X, y = make_circles(n_samples=1000, factor=0.3, noise=0.05, random_state=0)
    # Create an SVM classifier with the selected kernel
    clf = SVC(kernel=kernel)
    # Fit the classifier to the data
    clf.fit(X, y)
     # Define the decision boundary levels
    xx, yy, Z = define_decision_boundary_levels(X, clf)
    # Plot the decision boundary
    plot_decision_boundary(X, y, clf, xx, yy, Z, kernel)


def find_best_kernel_moons(kernel):
    X, y = make_moons(n_samples=1000, noise=0.05, random_state=0)
    # Create an SVM classifier with the selected kernel
    clf = SVC(kernel=kernel)
    # Fit the classifier to the data
    clf.fit(X, y)
    xx, yy, Z = define_decision_boundary_levels(X, clf)
    # Plot the decision boundary
    plot_decision_boundary(X, y, clf, xx, yy, Z, kernel)

#for i in range (len(kernels)):
    #find_best_kernel_circles(kernels[i])

# Best kernel for both make_moons and make_cirlces solution appears to be the rbf kernel
# The decision boundary in both these solutions acts as a hyperplane that separates the classes
# into different categories in the dataset.
# in the example of the circle dataset, the data was distributed differently than the half moons  
# data and therefore the decision boundary will be different in shape in both examples.
# the type of kernel influences the shape and flexibility of the decision boundary,
# as the graphs showed linear kernel types assume that the data is linearly separable
# and wasnt the best fit whereas the rbf kernel was able to handle datasets that were more complex in their distribution

# Create MLP model
# source https://machinelearninggeek.com/multi-layer-perceptron-neural-network-using-python/
def train_mlp(X, y):
    # adjusted the hidden layer size to ensure that the model wasnt overfitting or underfitting
    mlp = MLPClassifier(hidden_layer_sizes=(4,2), max_iter=1000, random_state=5, learning_rate_init=0.01)
    mlp.fit(X, y)
    return mlp

# determine accuracy for mlp model on circles dataset
# and plot the decision boundary
def mlp_circles_dataset():
    X_circles, y_circles = make_circles(n_samples=900, factor=0.3, noise=0.05, random_state=5)
    mlp_circles = train_mlp(X_circles, y_circles)
    y_pred = mlp_circles.predict(X_circles)
    circles_accuracy = accuracy_score(y_circles, y_pred)
    print("Accuracy on the circles dataset:", circles_accuracy)

    # Visualize the decision boundary
    xx, yy = np.meshgrid(np.linspace(X_circles[:, 0].min(), X_circles[:, 0].max(), 100),
                         np.linspace(X_circles[:, 1].min(), X_circles[:, 1].max(), 100))
    Z = mlp_circles.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, edgecolors='k', cmap=plt.cm.RdBu)
    plt.xlabel('X Axis Value')
    plt.ylabel('Y Axis Value')
    plt.title('MLP Decision Boundary')
    plt.show()

# determine accuracy for mlp trained on moons dataset and plot decision boundary
def mlp_moons_dataset():
    X_moons, y_moons = make_moons(n_samples=900, noise=0.05, random_state=5)
    # since the circles dataset is more challenging to classify
    # im adjusting the mlp parameters to a more complex architecture to ensure a higher accuracy score
    mlp = MLPClassifier(hidden_layer_sizes=(3,8), max_iter=1000, random_state=5, learning_rate_init=0.01)
    mlp.fit(X_moons, y_moons)
    y_pred = mlp.predict(X_moons)
    moons_accuracy = accuracy_score(y_moons, y_pred)
    print("Accuracy on the moon dataset:", moons_accuracy)
    # Visualize the decision boundary
    xx, yy = np.meshgrid(np.linspace(X_moons[:, 0].min(), X_moons[:, 0].max(), 100),
                         np.linspace(X_moons[:, 1].min(), X_moons[:, 1].max(), 100))
    Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, edgecolors='k', cmap=plt.cm.RdBu)
    plt.xlabel('X Axis Value')
    plt.ylabel('Y Axis Value')
    plt.title('MLP Decision Boundary')
    plt.show()

mlp_circles_dataset()
mlp_moons_dataset()
