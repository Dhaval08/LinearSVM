import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
from svmutil import *
import time

# **************************Part A *************************************

def calculateMSE(parameters, data, target):

    total = 0

    for i in range (0, len(target)):
        total = total + math.pow((np.dot(parameters[:,0], data[i,:])) - target[i], 2)

    return total/len(target)

#def g1Function(target, size):

def g1Function(target, size):
    bias_list= []
    variance_list = []
    mean_squared_error = []

    for i in range(0, len(target)):

        MSESum = 0

        predicted = np.ones(shape=(size,1))

        for k in range (0, len(target[i])):
            MSESum = MSESum + math.pow(1 - target[i][k,0], 2)

        mean_squared_error.append(MSESum/len(target))

        mean_predicted = np.mean(predicted)

        variance_sum = 0

        for k in range(0, len(predicted)):
            variance_sum = variance_sum + math.pow(predicted[k] - mean_predicted , 2)

        variance_list.append(variance_sum/len(predicted))

        total = 0.0
        for j in range (0, len(target[i])):
            total = total + math.pow(predicted[j] - target[i][j], 2)

        bias_list.append(total/len(target[i]))   # Storing bias for each data set

    bias = sum(bias_list)/len(bias_list)
    variance = sum(variance_list)/len(variance_list)

    print 'Bias^2 for g1 is', bias
    print 'Variance for g1 is', variance

    plt.hist(mean_squared_error, bins=10)
    plt.title('g1')
    plt.savefig('G1')
    plt.show()


def g2Function(target, size):
    g2_data = np.ones(shape=(size,1))
    split_g2_data = np.split(g2_data, len(target))
    g1_weights = np.zeros(shape=(1,1))
    mean_squared_error = []
    bias_list = []
    variance_list = []

    for i in range(0, len(target)):
        g1_weights = linearRegression(split_g2_data[i], target[i], g1_weights)

        mean_squared_error.append(calculateMSE(g1_weights, split_g2_data[i], target[i]))

        predicted = np.dot(g1_weights[:,0], np.transpose(split_g2_data[i]))

        mean_predicted = np.mean(predicted)

        variance_sum = 0

        for k in range(0, len(predicted)):
            variance_sum = variance_sum + math.pow(predicted[k] - mean_predicted , 2)

        variance_list.append(variance_sum/len(predicted))

        total = 0.0
        for j in range (0, len(target[i])):
            total = total + math.pow((np.dot(g1_weights[:,0], split_g2_data[i][j])) - target[i][j], 2)

        bias_list.append(total/len(target[i]))   # Storing bias for each data set

    bias = sum(bias_list)/len(bias_list)
    variance = sum(variance_list)/len(variance_list)

    print 'Bias^2 for g2 is', bias
    print 'Variance for g2 is', variance

    plt.hist(mean_squared_error, bins=10)
    plt.title('g2')
    plt.savefig('G2')

    plt.show()

def g3Function(data, target, size):
    mean_squared_error = []
    bias_list = []
    variance_list = []

    for i in range(0, len(target)):

        g2_data = np.ones(shape=(size,1))
        g2_data = np.concatenate((g2_data, data[i]), axis = 1)

        g1_weights = np.zeros(shape=(2,1))

        g1_weights = linearRegression(g2_data, target[i], g1_weights)
        mean_squared_error.append(calculateMSE(g1_weights, g2_data, target[i]))

        predicted = np.dot(g1_weights[:,0], np.transpose(g2_data))

        mean_predicted = np.mean(predicted)

        variance_sum = 0

        for k in range(0, len(predicted)):
            variance_sum = variance_sum + math.pow(predicted[k] - mean_predicted , 2)

        variance_list.append(variance_sum/len(predicted))

        total = 0.0
        for j in range (0, len(target[i])):
            total = total + math.pow((np.dot(g1_weights[:,0], g2_data[j,:])) - target[i][j], 2)

        bias_list.append(total/len(target[i]))   # Storing bias for each data set

    bias = sum(bias_list)/len(bias_list)
    variance = sum(variance_list)/len(variance_list)

    print 'Bias^2 for g3 is', bias
    print 'Variance for g3 is', variance

    plt.hist(mean_squared_error, bins=10)
    plt.title('g3')
    plt.savefig('G3')

    plt.show()

def g4Function(data, target, size):
    mean_squared_error = []
    bias_list = []
    variance_list = []

    for i in range(0, len(target)):

        g2_data = np.ones(shape=(size,1))
        g2_data = np.concatenate((g2_data, data[i]), axis = 1)
        g2_data = np.concatenate((g2_data, np.power(data[i], 2)), axis=1)

        g1_weights = np.zeros(shape=(2,1))

        g1_weights = linearRegression(g2_data, target[i], g1_weights)
        mean_squared_error.append(calculateMSE(g1_weights, g2_data, target[i]))

        predicted = np.dot(g1_weights[:,0], np.transpose(g2_data))

        mean_predicted = np.mean(predicted)

        variance_sum = 0

        for k in range(0, len(predicted)):
            variance_sum = variance_sum + math.pow(predicted[k] - mean_predicted , 2)

        variance_list.append(variance_sum/len(predicted))

        total = 0.0
        for j in range (0, len(target[i])):
            total = total + math.pow((np.dot(g1_weights[:,0], g2_data[j,:])) - target[i][j], 2)

        bias_list.append(total/len(target[i]))   # Storing bias for each data set

    bias = sum(bias_list)/len(bias_list)
    variance = sum(variance_list)/len(variance_list)

    print 'Bias^2 for g4 is', bias
    print 'Variance for g4 is', variance

    plt.hist(mean_squared_error, bins=10)
    plt.title('g4')
    plt.savefig('G4')

    plt.show()

def g5Function(data, target, size):
    mean_squared_error = []
    bias_list = []
    variance_list = []

    for i in range(0, len(target)):

        g2_data = np.ones(shape=(size,1))
        g2_data = np.concatenate((g2_data, data[i]), axis = 1)
        g2_data = np.concatenate((g2_data, np.power(data[i], 2)), axis=1)
        g2_data = np.concatenate((g2_data, np.power(data[i], 3)), axis=1)

        g1_weights = np.zeros(shape=(2,1))

        g1_weights = linearRegression(g2_data, target[i], g1_weights)
        mean_squared_error.append(calculateMSE(g1_weights, g2_data, target[i]))

        predicted = np.dot(g1_weights[:,0], np.transpose(g2_data))

        mean_predicted = np.mean(predicted)

        variance_sum = 0

        for k in range(0, len(predicted)):
            variance_sum = variance_sum + math.pow(predicted[k] - mean_predicted , 2)

        variance_list.append(variance_sum/len(predicted))

        total = 0.0
        for j in range (0, len(target[i])):
            total = total + math.pow((np.dot(g1_weights[:,0], g2_data[j,:])) - target[i][j], 2)

        bias_list.append(total/len(target[i]))   # Storing bias for each data set

    bias = sum(bias_list)/len(bias_list)
    variance = sum(variance_list)/len(variance_list)

    print 'Bias^2 for g5 is', bias
    print 'Variance for g5 is', variance

    plt.hist(mean_squared_error, bins=10)
    plt.title('g5')
    plt.savefig('G5')
    plt.show()

def g6Function(data, target, size):
    mean_squared_error = []
    bias_list = []
    variance_list = []

    for i in range(0, len(target)):

        g2_data = np.ones(shape=(size,1))
        g2_data = np.concatenate((g2_data, data[i]), axis = 1)
        g2_data = np.concatenate((g2_data, np.power(data[i], 2)), axis=1)
        g2_data = np.concatenate((g2_data, np.power(data[i], 3)), axis=1)
        g2_data = np.concatenate((g2_data, np.power(data[i], 4)), axis=1)

        g1_weights = np.zeros(shape=(2,1))

        g1_weights = linearRegression(g2_data, target[i], g1_weights)
        mean_squared_error.append(calculateMSE(g1_weights, g2_data, target[i]))

        predicted = np.dot(g1_weights[:,0], np.transpose(g2_data))

        mean_predicted = np.mean(predicted)

        variance_sum = 0

        for k in range(0, len(predicted)):
            variance_sum = variance_sum + math.pow(predicted[k] - mean_predicted , 2)

        variance_list.append(variance_sum/len(predicted))

        total = 0.0
        for j in range (0, len(target[i])):
            total = total + math.pow((np.dot(g1_weights[:,0], g2_data[j,:])) - target[i][j], 2)

        bias_list.append(total/len(target[i]))  # Storing bias for each data set

    bias = sum(bias_list)/len(bias_list)
    variance = sum(variance_list)/len(variance_list)

    print 'Bias^2 for g6 is', bias
    print 'Variance for g6 is', variance

    plt.hist(mean_squared_error, bins=10)
    plt.title('g6')
    plt.savefig('G6')
    plt.show()

def linearRegression(train_data, train_target, linear_parameters):
    first_term = np.dot(np.transpose(train_data), train_data)
    linear_parameters = np.dot(np.linalg.pinv(first_term), np.dot(np.transpose(train_data), train_target))

    return linear_parameters

def regularizedRegression(data, target, parameters, lamda):
    first_term = np.dot(np.transpose(data),data) + np.dot(lamda, np.identity(3))
    parameters = np.dot(np.linalg.pinv(first_term), np.dot(np.transpose(data), target))

    return parameters

def hFunction(data, target, size, lamda):
    mean_squared_error = []
    bias_list = []
    variance_list = []

    for i in range(0, len(target)):

        g2_data = np.ones(shape=(size,1))
        g2_data = np.concatenate((g2_data, data[i]), axis = 1)
        g2_data = np.concatenate((g2_data, np.power(data[i], 2)), axis=1)


        g1_weights = np.zeros(shape=(2,1))

        g1_weights = regularizedRegression(g2_data, target[i], g1_weights, lamda)
        mean_squared_error.append(calculateMSE(g1_weights, g2_data, target[i]))

        predicted = np.dot(g1_weights[:,0], np.transpose(g2_data))

        mean_predicted = np.mean(predicted)

        variance_sum = 0

        for k in range(0, len(predicted)):
            variance_sum = variance_sum + math.pow(predicted[k] - mean_predicted , 2)

        variance_list.append(variance_sum/len(predicted))

        total = 0.0
        for j in range (0, len(target[i])):
            total = total + math.pow((np.dot(g1_weights[:,0], g2_data[j,:])) - target[i][j], 2)

        bias_list.append(total/len(target[i]))   # Storing bias for each data set

    bias = sum(bias_list)/len(bias_list)
    variance = sum(variance_list)/len(variance_list)

    print 'Bias^2 for h function and lambda=', lamda, 'is', bias
    print 'Variance for h function and lambda=', lamda, 'is', variance

    plt.hist(mean_squared_error, bins=10)
    plt.show()

uniform_samples = np.empty(shape=(1000,1))

uniform_samples[:,0] = np.random.uniform(-1, 1, 1000)

target = np.empty(shape=(1000,1))

for i in range (0, 1000):
    target[i] = 2*math.pow(uniform_samples[i,0],2) + np.random.normal(0,0.1)

split_data = np.split(uniform_samples, 100)
split_target = np.split(target, 100)

print '-----------For 10 samples in each data set-----------\n'

g1Function(split_target, 1000)
g2Function(split_target, 1000)

g3Function(split_data, split_target, 10)
g4Function(split_data, split_target, 10)
g5Function(split_data, split_target, 10)

g6Function(split_data, split_target, 10)

print '\n---------For 100 samples in each data set----------\n'

uniform_samples = np.empty(shape=(10000,1))

uniform_samples[:,0] = np.random.uniform(-1, 1, 10000)

target = np.empty(shape=(10000,1))

for i in range (0, 10000):
    target[i] = 2*math.pow(uniform_samples[i,0],2) + np.random.normal(0,0.1)

split_data = np.split(uniform_samples, 100)
split_target = np.split(target, 100)

g1Function(split_target, 10000)
g2Function(split_target, 10000)

g3Function(split_data, split_target, 100)
g4Function(split_data, split_target, 100)
g5Function(split_data, split_target, 100)
g6Function(split_data, split_target, 100)

print '\n---------For h function-------------\n'

for lamda in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]:
    hFunction(split_data, split_target, 100, lamda)

# **************************Part B *************************************

# -------------------------- Data Pre-processing--------------------------
def preProcess(train):
    feature_list = [1, 6, 7, 13, 14, 25, 28]

    column_count = train.shape[1]

    transformed_data = np.empty(shape=(2000,0))

    for i in range(0,column_count):
        if i in feature_list:
            first = np.zeros(shape=(2000,1))
            second = np.zeros(shape=(2000,1))
            third = np.zeros(shape=(2000,1))
            for j in range(0,2000):
                if(train[j,i] == -1):
                    first[j,0] = 1
                if(train[j,i] == 1):
                    second[j,0] = 1
                if(train[j,i] == 0):
                    third[j,0] = 1

            transformed_data = np.concatenate((transformed_data, first), axis=1)
            transformed_data = np.concatenate((transformed_data, second), axis=1)
            transformed_data = np.concatenate((transformed_data, third), axis=1)

        else:

            if(-1 not in train[:,i]):
                column = np.empty(shape=(2000,1))
                column[:,0] = train[:,i]
                transformed_data = np.concatenate((transformed_data, column), axis=1)

            else:
                first = np.zeros(shape=(2000,1))
                for j in range(0,2000):
                    if(train[j,i] == 1):
                        first[j,0] = 1

                transformed_data = np.concatenate((transformed_data, first), axis=1)

    return transformed_data
# -------------------------Loading the data-----------------------------

train_data = sio.loadmat('phishing-train.mat')
test_data = sio.loadmat('phishing-test.mat')


phishing_train = train_data['features']
train_target = train_data['label']

phishing_test = test_data['features']
test_target = test_data['label']

phishing_train = np.asarray(phishing_train)
phishing_test = np.asarray(phishing_test)

processed_data = preProcess(phishing_train)
processed_test = preProcess(phishing_test)

train_target = train_target.tolist()
test_target = test_target.tolist()

processed_data = processed_data.tolist()
processed_test = processed_test.tolist()

max_accuracy = float("-inf")
optimal_c = 0

print '\n---------Linear SVM---------\n'

for i in (-6, -5, -4, -3, -2, -1, 0, 1, 2):
    C = math.pow(4,i)


    start_time = time.time()

    m = svm_train(train_target[0], processed_data, '-c %f -v 3 -q' %C)

    average_time = (time.time() - start_time)/3

    #print 'Accuracy for C =', C,'is', m
    print 'Average time for C=',C,'is =', average_time

    if(m>max_accuracy):
        max_accuracy = m
        optimal_c = C

m = svm_train(train_target[0], processed_data, '-c %f -q' %optimal_c)

svm_predict(train_target[0], processed_data, m)
svm_predict(test_target[0], processed_test, m)

max_polynomial_accuracy = float("-inf")
optimal_polynomial_degree = 0
optimal_polynomial_C = 0

print '\n---------Polynomial Kernel SVM----------\n'

for i in (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7):

    C = math.pow(4, i)
    for degree in (1, 2, 3):

        start_time = time.time()

        m = svm_train(train_target[0], processed_data, '-c {} -v 3 -d {} -q'.format(C, degree))

        average_time = (time.time() - start_time)/3

        print 'Average training time for C=',C,'degree =', degree,'is ',average_time

        #print 'For value of C=',C,'and degree=',degree,'accuracy is', m

        if m > max_polynomial_accuracy:
            max_polynomial_accuracy = m
            optimal_polynomial_degree = degree
            optimal_polynomial_C = C

#print 'Maximum cross validation accuracy for polynomial kernel', max_polynomial_accuracy

max_gamma_accuracy = float("-inf")
optimal_gamma = 0
optimal_gamma_C = 0

for i in (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7):
    C = math.pow(4, i)

    for j in range(-7,0):

        gamma = math.pow(4, j)

        start_time = time.time()

        m = svm_train(train_target[0], processed_data, '-c {} -v 3 -g {} -q'.format(C, gamma))

        average_time = (time.time()- start_time)/3

        print 'For value of C=',C,'and gamma=',gamma,'average training time is', average_time

        if m > max_gamma_accuracy:
            max_gamma_accuracy = m
            optimal_gamma = gamma
            optimal_gamma_C = C

#print 'RBF Kernel gives better accuracy'

#print("Maximum gamma accuracy",max_gamma_accuracy)
#print(optimal_gamma)
#print(optimal_gamma_C)

