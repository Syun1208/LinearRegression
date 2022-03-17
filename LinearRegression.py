import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

class LinearRegression:
    def __init__(self, x, y, theta1, theta2, epochs, learning_rate):
        self.x = x
        self.y = y
        self.theta1 = theta1
        self.theta2 = theta2
        self.learning_rate = learning_rate
        self.epochs = epochs
    def gradientDecent(self):
        d_theta1 = 0
        d_theta2 = 0
        for i in range(0, self.epochs):
            d_theta1 += self.learning_rate * 2 * (self.theta1 * self.x[i] + self.theta2 - self.y[i]) * self.x[i]
            d_theta2 += self.learning_rate * 2 * (self.theta1 * self.x[i] + self.theta2 - self.y[i])

        self.theta1 = self.theta1 - d_theta1 / self.epochs
        self.theta2 = self.theta2 - d_theta2 / self.epochs

        return self.theta1, self.theta2

if __name__ == '__main__':
    x = []
    y = []
    epochs = 10
    for j in range(0, epochs):
        x.append(np.random.randint(-50, 50))
        y.append(np.random.randint(-50, 50))
    theta_1 = -3
    theta_2 = 10
    learning_rate = 1e-5
    LR = LinearRegression(x, y, theta_1, theta_2, epochs, learning_rate)
    for i in range(0, epochs):
        result_theta1, result_theta2 = LR.gradientDecent()
        plt.plot(x, y, 'ro')
        plt.plot(x, np.asarray(result_theta1) * x + np.asarray(result_theta2))
        plt.show()
        clear_output(wait=True)
    print("Theta1_gradienDescent: {}".format(result_theta1))
    print("Theta2_gradienDescent: {}".format(result_theta2))
