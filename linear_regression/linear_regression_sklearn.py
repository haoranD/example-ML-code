import numpy as np

import matplotlib.pyplot as plt

m = 30
theta0_true = 2
theta1_true = 0.5
X = np.linspace(-1,1,m)

rand_noise_mu = 0
rand_noise_std = 0.1

rand_noise = np.random.normal(rand_noise_mu, rand_noise_std, m)
Y = theta0_true + theta1_true * X + rand_noise

from sklearn import linear_model
X = np.reshape(X, (-1, 1))

regr = linear_model.LinearRegression()
regr.fit(X, Y)

theta0 = regr.intercept_
theta1 = regr.coef_
print('theta1: \n', theta0)
print('theta0: \n', theta1)

plt.scatter(X, Y)
plt.plot(X, theta0 + theta1*X)
plt.show()