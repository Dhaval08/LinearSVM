import scipy.io as sio
import numpy as np
# -------------------------- Data Pre-processing--------------------------
def preProcess(train):
    feature_list = [2]

    column_count = train.shape[1]

    transformed_data = np.empty(shape=(2000,0))

    for i in range(0,column_count):
        if i+1 in feature_list:
            first = np.zeros(shape=(2000,1))
            second = np.zeros(shape=(2000,1))
            third = np.zeros(shape=(2000,1))
            for j in range(0,2000):
                if(train[j,i] == -1):
                    first[j,0] = 1
                elif(train[j,i] == 1):
                    second[j,0] = 1
                elif(train[j,i] == 0):
                    third[j,0] == 1

            print(first)

            transformed_data = np.concatenate((transformed_data, first), axis=1)
            transformed_data = np.concatenate((transformed_data, second), axis=1)
            transformed_data = np.concatenate((transformed_data, third), axis=1)

        else:
            column = np.empty(shape=(2000,1))
            column[:,0] = train[:,i]
            transformed_data = np.concatenate((transformed_data, column), axis=1)


    print(transformed_data)
    return transformed_data
# -------------------------Loading the data-----------------------------

train_data = sio.loadmat('phishing-train.mat')
phishing_train = np.empty(shape=(2000,30))

phishing_train = train_data['features']

phishing_train = np.asarray(phishing_train)

processed_data = preProcess(phishing_train)
