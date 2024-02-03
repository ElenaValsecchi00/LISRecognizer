import os 
import json
import numpy as np
from scipy.spatial import distance
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

#create a dict with all the points for every label
def construct_dict():
    dataset_dict = {}

    #scans landmarks
    for dir in os.scandir('LISRecognition/landmarks'):
        key = dir.name
        dataset_dict[key] = []
        for file in os.scandir(dir.path):
            with open(file.path, 'r') as f:
                file_content = list(json.load(f))
            dataset_dict[key].append(file_content)
        if dataset_dict[key] == []:
            del dataset_dict[key]

    #shuffle and cut data
    for key in dataset_dict:
        shuffle(dataset_dict[key])
        dataset_dict[key] = dataset_dict[key][:40]
    return dataset_dict

#calculate distance vectors
def calculate_distances(points):
    distances = []

    #calculate euclidead distance between a point and its subsequent
    for point in range(len(points)-1):
        dist = distance.euclidean(points[point], points[point+1])
        distances.append(dist)

    return distances

#create final and lables
def create_distance_vector(dataset_dict):
    distance_vector = []
    lables = []
    for key in dataset_dict:
        for landmark_collection in dataset_dict[key]:
            distances = calculate_distances(landmark_collection)
            lables.append(key)
            min_val = np.min(distances)
            max_val = np.max(distances)
            distances = (distances - min_val) / (max_val - min_val)
            distance_vector.append(distances)

    return np.array(distance_vector),np.array(lables)

def main():
    dataset_dict = construct_dict()
    distance_vector, lables = create_distance_vector(dataset_dict)
    distance_vector = distance_vector.reshape(880,1,20)
    le = LabelEncoder()
    le.fit(lables)
    int_lables = le.transform(lables)
    int_lables = np_utils.to_categorical(int_lables)
    X_train, X_test, y_train, y_test = train_test_split(distance_vector, int_lables, test_size=3 / 10, train_size = 7/10, random_state=1127)
    return X_train, X_test, y_train, y_test

main()
