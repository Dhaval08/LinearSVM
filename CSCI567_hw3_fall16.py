import scipy.io as sio
import numpy as np
import math
from svmutil import *

# -------------------------- Data Pre-processing--------------------------
def preProcess(train):
    feature_list = [1, 6, 7, 13, 14, 15, 25, 28]

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
            column = np.empty(shape=(2000,1))
            column[:,0] = train[:,i]
            transformed_data = np.concatenate((transformed_data, column), axis=1)

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

for i in (-6, -5, -4, -3, -2, -1, 0, 1, 2):
    C = math.pow(4,i)

    m = svm_train(train_target[0], processed_data, '-c %f -v 3 -q' %C)

    if(m>max_accuracy):
        max_accuracy = m
        optimal_c = C


m = svm_train(train_target[0], processed_data, '-c %f -q' %optimal_c)

p_labs, p_acc, p_vals = svm_predict(test_target[0], processed_test, m)

print(p_acc)


max_kernel_accuracy = float("-inf")
optimal_degree = 0

for i in (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7):

    C = math.pow(4, i)
    for degree in [1, 2, 3]:
        m = svm_train(train_target[0], processed_data, '-c {} -v 3 -d {} -q'.format(C, degree))

    if m > max_kernel_accuracy:
        max_kernel_accuracy = m
        optimal_degree = degree
        optimal_kernel_C = C

print(optimal_kernel_C)
print(optimal_degree)