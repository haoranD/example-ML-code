import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# to create dataset
m = 30
theta0_true = 2
theta1_true = 0.5
X = np.linspace(-1,1,m)

rand_noise_mu = 0
rand_noise_std = 0.1

rand_noise = np.random.normal(rand_noise_mu, rand_noise_std, m)
Y = theta0_true + theta1_true * X + rand_noise

#print and plot the data
print('X',X)
print('Y',Y)

plt.scatter(X, Y)

#define the cost/loss function and gradient descent function

def cost_MSE(theta0,theta1,X,Y):
    hypothesis = theta0 + theta1*X
    m = len(X)
    return (1/(2.0*m))*((np.square(hypothesis-Y)).sum(axis=0))


def gradientDescent(theta0,theta1,X,Y,iterations,alpha):
    count = 1
    cost_log = np.array([])
    m = len(X)
    
    while(count <= iterations):
        hypothesis = theta0 + theta1*X
        theta0 = theta0 - alpha*(1.0/m)*((hypothesis-Y)).sum(axis=0)
        theta1 = theta1 - alpha*(1.0/m)*((hypothesis-Y)*X).sum(axis=0)
        cost_log = np.append(cost_log,cost_MSE(theta0,theta1,X,Y))
        count = count + 1
        
    plt.subplot(121)
    plt.plot(np.linspace(1,iterations,iterations,endpoint=True),cost_log)
    plt.title("Cost/Loss wrt iteration")
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost/Loss")
    
    
    plt.subplot(122)
    plt.scatter(X, Y)
    plt.plot(X, theta0 + theta1*X)
    plt.show()
    
    
    return theta0,theta1

alpha = 0.3
iterations = 20

Thetas_Init = np.random.rand(2)

plt.figure(figsize=(10, 4))

theta0,theta1 = gradientDescent(Thetas_Init[0],Thetas_Init[1], X, Y,iterations,alpha)
print('theta0:', theta0)
print('theta1:', theta1)



