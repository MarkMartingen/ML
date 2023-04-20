
import random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


''' I will use know the newton method, to minimize the J() cost function'''

''' It can be shown that the hessian of the logistic cost function is XDX', where ' means transpose and D 
is a diagonal matrix with entries equal to  sigmoid(X*Theta)(1- sigmoid(X*Theta)'''

# obtain data:
X = pd.read_csv('C:\\users\Marcin Basiuk\\Downloads\\house.csv')
y = X[['price_bin']]
#X = X[['sqft_living']]
X = X[['sqft_living', 'sqft_basement', 'bedrooms']]
tmp = copy.deepcopy(X.columns)
for c in X.columns:
    X[c] = (X[c] - X[c].mean()) / X[c].std()
X['intercept'] = 1
X = X[['intercept'] + list(tmp)]


# Initialize Theta again to be random:
Theta = pd.DataFrame([random.random() for j in range(X.shape[1])])
Theta = Theta.to_numpy()


def sigmoid(x):
    if type(x) == float:
        return (1 / (1 + math.exp(-x)))
    if type(x) == np.ndarray:
            result = []
            for i in range(len(x)):
                try:
                    result.append((1 / (1 + math.exp(-x[i]))))
                except OverflowError:
                    if x[i] >= 0:
                        result.append(0.0)
                    else:
                        result.append(1.0)
            return np.array(result).reshape(len(x),1)


def logit_likelyhood(X,y, Theta):
    #c = (logit(X.to_numpy(), Theta) - y.to_numpy())**2

    z = sigmoid(np.matmul(X.to_numpy(), Theta))
    c = np.log(z, where=z > 0)*y.to_numpy() + (np.ones((X.shape[0], 1)) -
                    y.to_numpy()) * np.log(np.ones((X.shape[0], 1)) - z, where=np.ones((X.shape[0], 1)) - z > 0)
    return c.sum()

def logit_cost_gradient(X, y, Theta):
    return np.matmul((sigmoid(np.matmul(X.to_numpy(), Theta)) - y.to_numpy()).transpose(), X.to_numpy()).reshape(len(Theta), 1)

def hessian(X, Theta):
    m = X.shape[0]
    D = sigmoid(np.matmul(X.to_numpy(), Theta))*(np.array([1]*m).reshape(m, 1) - sigmoid(np.matmul(X.to_numpy(), Theta)))
    D = np.diag(D.reshape(m,))
    return np.matmul(np.matmul(X.to_numpy().transpose(), D), X.to_numpy())

C = []
R = []
print('initial log likelyhood: ',  logit_likelyhood(X, y, Theta))
maxiter = 10
for i in range(maxiter):
    H = hessian(X, Theta)

    try:
        Theta = Theta - np.matmul(np.linalg.inv(H), logit_cost_gradient(X, y, Theta))
    except np.linalg.LinAlgError:
        print(H)
        break
    likelyhood = logit_likelyhood(X, y, Theta)
    C.append(likelyhood)
    print('iteration: ', i, '  likelyhood: ', likelyhood)
    R.append(i)

print(C)
print(Theta)
plt.plot(R, C)
plt.title(' Logarithm of likelyhood as a function of iterations')
plt.xlabel('number of iterations')
plt.ylabel('logarithm of likelyhood')
plt.show()



def predict(x):
    if x > 0.5:
        return 1
    else:
        return 0

def output_results(X, y, Theta):
    y_pred  = np.matmul(X.to_numpy(), Theta)
    y_pred  = np.array([predict(a) for a in y_pred[:, 0]])
    results = pd.DataFrame({'y_pred': y_pred.reshape(y_pred.shape[0], ), 'y_fact': y['price_bin'].to_numpy()})
    ''' COnfusion matrix :'''
    conf_matrix = pd.crosstab(results['y_fact'], results['y_pred'], rownames=['Actual'], colnames=['Predicted'])
    print(conf_matrix)

    accuracy = (conf_matrix.iloc[0, 0]+ conf_matrix.iloc[1, 1])/conf_matrix.sum().sum()
    precison = conf_matrix.iloc[1,1]/(conf_matrix.iloc[1,1] + conf_matrix.iloc[0,1])
    recall = conf_matrix.iloc[1,1]/(conf_matrix.iloc[1,1] + conf_matrix.iloc[1,0])

    print('accuracy: ', accuracy, '  all zeros accuracy: ', y[(y['price_bin'] == 0)].count()/y.count())
    print('precsision: ', precison)
    print('recall: ', recall)
    print('F1-score: ', 2*precison*recall/(precison + recall))

output_results(X, y, Theta)







