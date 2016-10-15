import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
from svmutil import *

# **************************Part A *************************************
'''
def calculateMSE(parameters, data, target):

    total = 0

    for i in range (0, len(target)):
        total = total + math.pow((np.dot(parameters[:,0], data[i,:])) - target[i], 2)

    return total/len(target)

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

    print bias,'\t \t', variance, '\n'

    plt.hist(mean_squared_error, bins=10)
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

    print bias,'\t \t', variance, '\n'


    plt.hist(mean_squared_error, bins=10)
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

    print bias,'\t \t', variance, '\n'


    plt.hist(mean_squared_error, bins=10)
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

    print bias,'\t \t', variance, '\n'


    plt.hist(mean_squared_error, bins=10)
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

    print bias,'\t \t', variance, '\n'


    plt.hist(mean_squared_error, bins=10)
    plt.show()

def linearRegression(train_data, train_target, linear_parameters):
    first_term = np.dot(np.transpose(train_data), train_data)
    linear_parameters = np.dot(np.linalg.pinv(first_term), np.dot(np.transpose(train_data), train_target))

    return linear_parameters

def regularizedRegression(data, target, parameters, lamda):
    first_term = np.dot(np.transpose(data),data) + np.dot(lamda, np.identity(3))
    parameters = np.dot(np.linalg.pinv(first_term), np.dot(np.transpose(data), target))

    return parameters

def hFunction(data, target, size):
    mean_squared_error = []
    bias_list = []
    variance_list = []

    for i in range(0, len(target)):

        g2_data = np.ones(shape=(size,1))
        g2_data = np.concatenate((g2_data, data[i]), axis = 1)
        g2_data = np.concatenate((g2_data, np.power(data[i], 2)), axis=1)


        g1_weights = np.zeros(shape=(2,1))

        g1_weights = regularizedRegression(g2_data, target[i], g1_weights, 0.001)
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

    print bias,'\t \t', variance, '\n'


    plt.hist(mean_squared_error, bins=10)
    plt.show()



uniform_samples = np.empty(shape=(1000,1))

uniform_samples[:,0] = np.random.uniform(-1, 1, 1000)

target = np.empty(shape=(1000,1))

for i in range (0, 1000):
    target[i] = 2*math.pow(uniform_samples[i,0],2) + np.random.normal(0,0.1)

split_data = np.split(uniform_samples, 100)
split_target = np.split(target, 100)


print 'For 10 samples in each data set'
print 'Bias \t \t \t \t \t Variance'

g2Function(split_target, 1000)

g3Function(split_data, split_target, 10)
g4Function(split_data, split_target, 10)
g5Function(split_data, split_target, 10)

g6Function(split_data, split_target, 10)

print 'For 100 samples in each data set'
print 'Bias \t \t \t \t \t Variance'


uniform_samples = np.empty(shape=(10000,1))

uniform_samples[:,0] = np.random.uniform(-1, 1, 10000)

target = np.empty(shape=(10000,1))

for i in range (0, 10000):
    target[i] = 2*math.pow(uniform_samples[i,0],2) + np.random.normal(0,0.1)

split_data = np.split(uniform_samples, 100)
split_target = np.split(target, 100)

g2Function(split_target, 10000)

g3Function(split_data, split_target, 100)
g4Function(split_data, split_target, 100)
g5Function(split_data, split_target, 100)

g6Function(split_data, split_target, 100)


print 'For function h'
hFunction(split_data, split_target, 100)
'''


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

print(processed_data.shape)

processed_data = processed_data.tolist()
processed_test = processed_test.tolist()

max_accuracy = float("-inf")
optimal_c = 0

for i in (-6, -5, -4, -3, -2, -1, 0, 1, 2):
    C = math.pow(4,i)

    m = svm_train(train_target[0], processed_data, '-c %f -v 3 -q' %C)

    if(m>max_accuracy):
        max_accuracy = m
        optimal_c = C


print(max_accuracy)
print(optimal_c)

m = svm_train(train_target[0], processed_data, '-c %f -q' %optimal_c)


svm_predict(train_target[0], processed_data, m)
svm_predict(test_target[0], processed_test, m)

max_polynomial_accuracy = float("-inf")
optimal_polynomial_degree = 0
optimal_polynomial_C = 0

for i in (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7):

    C = math.pow(4, i)
    for degree in [1, 2, 3]:
        m = svm_train(train_target[0], processed_data, '-c {} -v 3 -d {} -q'.format(C, degree))

        if m > max_polynomial_accuracy:
            max_polynomial_accuracy = m
            optimal_polynomial_degree = degree
            optimal_polynomial_C = C

print(optimal_polynomial_C)
print(optimal_polynomial_degree)
print("Maximum polynomial kernel accuracy:", max_polynomial_accuracy)



max_gamma_accuracy = float("-inf")
optimal_gamma = 0
optimal_gamma_C = 0

for i in (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7):
    C = math.pow(4, i)

    for j in range(-7,0):

        gamma = math.pow(4, j)

        m = svm_train(train_target[0], processed_data, '-c {} -v 3 -g {} -q'.format(C, gamma))

        if m > max_gamma_accuracy:
            max_gamma_accuracy = m
            optimal_gamma = gamma
            optimal_gamma_C = C

print("Maximum gamma accuracy",max_gamma_accuracy)
print(optimal_gamma)
print(optimal_gamma_C)

