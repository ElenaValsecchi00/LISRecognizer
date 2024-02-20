import os 
import json
import numpy as np
from scipy.spatial import distance
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import DistanceMetric

#create a dict with all the points for every label
def construct_dict(path, cut):
    dataset_dict = {}

    #scans landmarks
    for dir in os.scandir(path):
        key = dir.name
        dataset_dict[key] = []
        if os.scandir(dir.path).__iter__().__next__().is_file():
            for file in os.scandir(dir.path):
                with open(file.path, 'r') as f:
                    file_content = list(json.load(f))
                dataset_dict[key].append(file_content[:21])
            if dataset_dict[key] == []:
                del dataset_dict[key]
        else:
            for subdir in os.scandir(dir.path):
                sequence = []
                for file in os.scandir(subdir.path):
                    with open(file.path, 'r') as f:
                        file_content = list(json.load(f))
                    sequence.append(file_content[:21])
                dataset_dict[key].append(sequence)
            if dataset_dict[key] == []:
                del dataset_dict[key]
    #shuffle and cut data
    for key in dataset_dict:
        shuffle(dataset_dict[key])
        dataset_dict[key] = dataset_dict[key][:cut]
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
def create_distance_vector(dataset_dict, moving):
    distance_vector = []
    lables = []
    if(type(dataset_dict)==dict): #if data are for training
        if(not moving): #if static gestures
            for key in dataset_dict:
                for landmark_collection in dataset_dict[key]:
                    distances = calculate_distances(landmark_collection)
                    lables.append(key)
                    min_val = np.min(distances)
                    max_val = np.max(distances)
                    distances = (distances - min_val) / (max_val - min_val)
                    distance_vector.append(distances)
            #np and reshaping
            distance_vector = np.array(distance_vector)
            distance_vector = distance_vector.reshape(len(distance_vector),1,20)
            lables = np.array(lables)
            le = LabelEncoder() #trasform lables into integers
            le.fit(lables)
            int_lables = le.transform(lables)
            int_lables = to_categorical(int_lables)
            return distance_vector,int_lables
        else: #if gesture are moving
            for key in dataset_dict:
                for sequence in dataset_dict[key]:
                    lables.append(key)
                    distance_subvector = []
                    for landmark_collection in sequence:
                        distances = calculate_distances(landmark_collection)
                        min_val = np.min(distances)
                        max_val = np.max(distances)
                        distances = (distances - min_val) / (max_val - min_val)
                        distance_subvector.append(distances)
                    distance_vector.append(distance_subvector)
            distance_vector = np.array(distance_vector)
            distance_vector = distance_vector.reshape(len(distance_vector),1,6*20)
            lables = np.array(lables)
            le = LabelEncoder() #trasform lables into integers
            le.fit(lables)
            int_lables = le.transform(lables)
            int_lables = to_categorical(int_lables)
            return distance_vector,int_lables
    ##when you submit a single frame or sequence of landmarks
    else:
        if(not moving):
            distances = calculate_distances(dataset_dict)
            min_val = np.min(distances)
            max_val = np.max(distances)
            distances = (distances - min_val) / (max_val - min_val)
            distance_vector.append(distances)
            distance_vector = np.array(distance_vector)
            distance_vector = distance_vector.reshape(len(distance_vector),1,20)
            return distance_vector
        else:
            for sequence in dataset_dict:
                distance_subvector = []
                for seq in dataset_dict:
                    distances = calculate_distances(seq)
                    min_val = np.min(distances)
                    max_val = np.max(distances)
                    distances = (distances - min_val) / (max_val - min_val)
                    distance_subvector.append(distances)
                distance_vector.append(distance_subvector)
            distance_vector = np.array(distance_vector)
            distance_vector = distance_vector.reshape(len(distance_vector),1,6*20)
            return distance_vector



def main():
    dataset_dict = construct_dict('landmarks', 80)
    distance_vector, int_lables = create_distance_vector(dataset_dict, False)
    X_train, X_test, y_train, y_test = train_test_split(distance_vector, int_lables, test_size=3 / 10, train_size = 7/10, random_state=1127)
    return X_train, X_test, y_train, y_test
