import random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

## Some random gaussian data:

centers = [[0, 0], [3, 3], [-2, -2], [6, 1], [-4, 2]]
sizes = [60, 100, 77, 60, 49]
var = 3
cov = [[var, 0.3], [0.3, var]]

label = 0
for mi in centers:
    try:
        y = np.vstack((y, label * np.ones((sizes[centers.index(mi)], 1))))
        X = np.vstack((X, np.random.multivariate_normal(mi, cov, sizes[centers.index(mi)])))
    except:
        y = label * np.ones((sizes[centers.index(mi)], 1))
        X = np.random.multivariate_normal(mi, cov, sizes[centers.index(mi)])
    label += 1


plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


## Make data frame, add intercept, shuffle and divide into training and test sets
P = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'y': y[:, 0]})
intercept = pd.DataFrame(np.ones((P.shape[0], 1)))
intercept.columns = ['intercept']

P = pd.concat([intercept, P], axis=1)

P = P.sample(frac=1)
Test = P.sample(frac=0.3)
Train = P.drop(Test.index)

## implement softmax regression:

def softmax(z):
    return (np.exp(z)/np.exp(z).sum())

def mylog(x):
    if x<=0:
        return 0
    else:
        return np.log(x)

def likelyhood(X, Theta, y, num_iter, alpha):
    loglikelyhood=[]

    for k in range(num_iter):
        Probs = X.dot(Theta).apply(softmax, axis=1)

        dummies = pd.get_dummies(y.astype(str))

        Probs = pd.concat([Probs, dummies], axis=1)

        classes = Theta.shape[1]
        residuals = Probs.iloc[:, 0:classes].to_numpy() - Probs.iloc[:, classes:].to_numpy()
        residuals = pd.DataFrame(residuals)

        for i in range(X.shape[0]):
            for j in range(classes):
                Theta[:, j] = Theta[:, j] - alpha*(residuals.iloc[i,j])*X.iloc[i, :].transpose()

        if k % 10 == 0:
            print ('iteration ', k, ' ..... out of ',num_iter )
        result = np.multiply(Probs.iloc[:, 0:classes], Probs.iloc[:, classes:])
        result = result.applymap(mylog)

        loglikelyhood.append(result.sum().sum())

    return loglikelyhood, Theta

X = Train[['intercept', 'x1', 'x2']]
y = Train[['y']]
num_classes = len(centers)
Theta = np.random.rand(X.shape[1], num_classes)

max_iter = 80
L = likelyhood(X, Theta, y, max_iter, 0.001)
print(L[0])
plt.plot(range(max_iter), L[0])
plt.show()


print('Classification accuracy on test set:')
Theta = L[1]
X = Test[['intercept', 'x1', 'x2']]
y = Test[['y']]
Probs = X.dot(Theta).apply(softmax, axis=1)


Prediction = Probs.apply(lambda x: np.argmax(x), axis =1)
Out = pd.concat([Prediction, y], axis=1)
Out.columns = ['y_pred', 'y']
Out = pd.crosstab(Out.y, Out.y_pred)
print(Out)


print('accuracy: ', Out.to_numpy().trace()/Out.sum().sum())
print(X.shape[0], ' - size of test sample')