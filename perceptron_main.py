import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from perceptron import SimplePerceptron

def fake_data(m): # m - no. of examples    
    m_half = int(m / 2)
    X_1 = np.random.rand(m, 1)
    X_2_down = np.random.rand(m_half, 1) * 0.4
    X_2_up = np.random.rand(m_half, 1) * 0.4 + 0.6
    y_2_down = np.ones(m_half, dtype=np.int8)
    y_2_up = -np.ones(m_half, dtype=np.int8)
    X = np.c_[X_1, np.r_[X_2_up, X_2_down]]
    y = np.r_[y_2_up, y_2_down]
    return X, y

def make_data(size = 1000):
    # Create empty list to store data
    data = []

    #Set seed
    np.random.seed(0)

    # Set the borders
    x1_borders = [0, 2*np.pi]
    x2_borders = [-1, 1]

    # Create the random dataset
    for i in range(size):
        x1 = np.random.uniform(x1_borders[0], x1_borders[1])
        x2 = np.random.uniform(x2_borders[0], x2_borders[1])
        if x1 != 0 and x2 != 0:
            if np.abs(np.sin(x1)) > np.abs(x2):
                y = -1
            else:
                y = 1
            data.append([x1, x2, y])

    # Convert to numpy array
    data = np.array(data)
    # return data

    # Normalize the data
    x1_norm = (data[:, 0] - x1_borders[0]) / (x1_borders[1] - x1_borders[0]) * 2 - 1
    x2_norm = (data[:, 1] - x2_borders[0]) / (x2_borders[1] - x2_borders[0]) * 2 - 1

    # Create the normalized dataset
    data_norm = np.column_stack((x1_norm, x2_norm))
    return data_norm, data[:, 2]

def generate_centers(m):

    #Set seed
    np.random.seed(0)
    
    # Set the borders
    borders = [-1, 1]
    data = []

    # Generate m random centers within borders
    for i in range(m):
        x1 = np.random.uniform(borders[0], borders[1])
        x2 = np.random.uniform(borders[0], borders[1])
        data.append([x1, x2])
    return np.array(data)

def gaussian_kernel(X, centers, sigma=1):
    # Calculate the distances
    distances = np.zeros((X.shape[0], centers.shape[0]))
    for i in range(X.shape[0]):
        for j in range(centers.shape[0]):
            distances[i, j] = np.exp(-((X[i, 0] - centers[j, 0])**2 + (X[i, 1] - centers[j, 1])**2) / (2 * sigma ** 2))
    return distances


if __name__ == '__main__':
    X, y = make_data(1000)
    # sigmas = np.arange(0.1, 1.1, 0.1)
    # counter = 1
    # scores_and_vars = []
    # for num_of_centers in range(20, 101, 10):
    #     for sigma in sigmas:
    #         for kmax in range(500, 5001, 100):
    #             centers = generate_centers(num_of_centers)
    #             distances = gaussian_kernel(X, centers, sigma)
                
    #             perceptron = SimplePerceptron(kmax=kmax, learning_rate=1)
    #             perceptron.fit(distances, y)
    #             preds = perceptron.predict(distances)
    #             score = perceptron.score(distances, y)
    #             scores_and_vars.append([score, num_of_centers, sigma, kmax])
    #             print(counter, "Done")
    #             counter += 1
                

    # scores_and_vars = np.array(scores_and_vars)
    # scores_and_vars = scores_and_vars[(-scores_and_vars[:, 0]).argsort()]
    # np.savetxt('scores_and_vars.csv',scores_and_vars, fmt = '%f', delimiter=",")

    centers = generate_centers(100)
    distances = gaussian_kernel(X, centers, 0.2)
                
    perceptron = SimplePerceptron(kmax=4000, learning_rate=1)
    perceptron.fit(distances, y)
    preds = perceptron.predict(distances)

    # Create color map
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    # Draw contour graph
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = perceptron.predict(gaussian_kernel(np.c_[xx.ravel(), yy.ravel()], centers, 0.2))
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.contour(xx, yy, Z, cmap=cmap_bold)
    for i in range(len(y)):
        if y[i] == preds[i]:
            if(y[i] == 1):
                plt.scatter(X[i, 0], X[i, 1], color='g')
            else:
                plt.scatter(X[i, 0], X[i, 1], color='r')
        else:
            if(y[i] == 1):
                plt.scatter(X[i, 0], X[i, 1], color='g', marker="x")
            else:
                plt.scatter(X[i, 0], X[i, 1], color='r', marker="x")
    plt.scatter(centers[:, 0], centers[:, 1], color="k")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig("cen=100_sigma=0.2_kmax=4000_score=0.99.jpg")    

    # plt.show()
    # filename = "cen=" + str(scores_and_vars[0, 1]) + "_sigma=" + str(scores_and_vars[0, 2]) + "_kmax=" + str(scores_and_vars[0, 3]) + "_score=" + str(scores_and_vars[0, 0]) + ".jpg"