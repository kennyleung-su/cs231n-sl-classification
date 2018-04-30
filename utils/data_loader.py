# TODO: Implement data loading and preprocessing.

import csv
import pandas as pd
import os


#returns dictionary with labels and the locations for the frames 
def gestures(*args, textfile= "", type_data="kinect"):
    data = pd.read_csv(textfile, sep=" ", header=None)
    data.columns = ["rgb", "kinect", "label"]
    gesture_labels ={}
    
    for number in args:
        labels =[]
        print(number)
        directory_labels = data.loc[data['label'] == number][type_data].tolist()
        for element in directory_labels:
            labels.append(element[:-4])
        gesture_labels[number] = labels
    return gesture_labels
gestures = gestures(5,19,38,37,8,29,213,241,18,92, textfile = 'train_list_2.txt', type_data = "kinect")
