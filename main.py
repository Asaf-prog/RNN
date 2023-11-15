import time
import json
from typing import List
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.metrics import get_accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

def euclidean_distance(valid, train_scaled):
    return distance.cdist(valid, train_scaled, 'euclidean')
class RadiusNNClassifier():
    def __init__(self, r):
        self.r = r

    def predict(self, vector_test):
        self.vector_test = vector_test
        self.m_test, self.n = vector_test.shape

        distance_matrix = euclidean_distance(vector_test, self.X_train)#creating the distance matrix
        neighbors = distance_matrix <= self.r
        label_predict = []
        for i in range(self.m_test):#for each row in the test data
            if np.any(neighbors[i]):
                label_predict.append(mode(self.Y_train[neighbors[i]])[0][0])#append the mode of the neighbors if there is a neighbor
            else:
                label_predict.append(self.Y_train[i-1])#append the previous label if there is no neighbor

        return label_predict


    def fit(self, vector_train, label_train):
        self.X_train = vector_train#saving the training data
        self.m, self.n = vector_train.shape#saving the shape of the training data
        self.Y_train = label_train#saving the training labels

def classify_with_NNR(data_trn, data_vld, data_tst) -> List:
    print(f'Starting classification with {data_trn}, {data_vld}, and {data_tst}')

    train = pd.read_csv(data_trn)
    train_vector = train.iloc[:, :-1]#extracting the training vector and labels
    label_train = train.iloc[:, -1:].values
    valid = pd.read_csv(data_vld)
    valid_vector = valid.iloc[:, :-1].values
    label_valid = valid.iloc[:, -1:].values
    test = pd.read_csv(data_tst)
    need_to_predict_values = test.iloc[:, :-1]


    scaler = StandardScaler()  # scaling the data using standard scaler
    vector_valid_after_scale = scaler.fit_transform(valid_vector)
    vector_train_after_scale = scaler.fit_transform(train_vector)
    vector_test_after_scale = scaler.fit_transform(need_to_predict_values)

    radius = find_radius(vector_train_after_scale, label_train, vector_valid_after_scale, label_valid)#calc radius

    model = RadiusNNClassifier(r=radius)#creating the model
    model.fit(vector_train_after_scale, label_train)
    result = model.predict(vector_test_after_scale)
    predictions = list(result)
    return predictions #creating the predictions list

def find_radius(train_vector, label_train, vector_valid, label_valid):
    distance_matrix = euclidean_distance(vector_valid, train_vector)
    radius_mean = distance_matrix.mean()
    current_radius = radius_mean
    optimal_radius = current_accuracy = optimal_accuracy = 0

    for i in range(100):

        model = RadiusNNClassifier(r=current_radius)
        model.fit(train_vector, label_train)
        predict = model.predict(vector_valid)
        current_accuracy = get_accuracy_score(label_valid, predict)#calc the accuracy

        if current_accuracy >= optimal_accuracy:#kind of early stopping :)
            optimal_accuracy = current_accuracy
            optimal_radius = current_radius
        else:
            break

        current_radius = distance_matrix[distance_matrix <= current_radius].mean()#calc the new radius and return the best

    return optimal_radius

if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  config['data_file_test'])

    df = pd.read_csv(config['data_file_test'])
    labels = df['class'].values

    if not predicted:
        predicted = list(range(len(labels)))

    assert len(labels) == len(predicted)
    print(f'Test set classification accuracy: {get_accuracy_score(labels, predicted)}')
