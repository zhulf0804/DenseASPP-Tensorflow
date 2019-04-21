# coding=utf-8
from __future__ import print_function
from __future__ import division

# Reference: https://raw.githubusercontent.com/kwotsin/TensorFlow-ENet/master/get_class_weights.py

import numpy as np
import os
from scipy.misc import imread
import ast

image_dir = "/Users/zhulf/data/CamVid/trainannot"
image_dir_2 = "/Users/zhulf/data/CamVid/valannot"
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')]
image_files_2 = [os.path.join(image_dir_2, file) for file in os.listdir(image_dir_2) if file.endswith('.png')]
image_files += image_files_2

def ENet_weighing(image_files=image_files, num_classes=12):
    '''
    The custom class weighing function as seen in the ENet paper.

    INPUTS:
    - image_files(list): a list of image_filenames which element can be read immediately

    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    #initialize dictionary with all 0
    label_to_frequency = {}
    for i in xrange(num_classes):
        label_to_frequency[i] = 0

    for n in xrange(len(image_files)):
        image = imread(image_files[n])

        #For each label in each image, sum up the frequency of the label and add it to label_to_frequency dict
        for i in xrange(num_classes):
            #print(np.unique(image))
            class_mask = np.equal(image, i)
            class_mask = class_mask.astype(np.float32)
            class_frequency = np.sum(class_mask)

            label_to_frequency[i] += class_frequency

    #perform the weighing function label-wise and append the label's class weights to class_weights
    class_weights = []
    total_frequency = sum(label_to_frequency.values())
    for label, frequency in label_to_frequency.items():
        class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
        class_weights.append(round(class_weight, 3))

    #Set the last class_weight to 0.0
    #class_weights[-1] = 0.0

    return class_weights

def median_frequency_balancing(image_files=image_files, num_classes=12):
    '''
    Perform median frequency balancing on the image files, given by the formula:
    f = Median_freq_c / total_freq_c

    where median_freq_c is the median frequency of the class for all pixels of C that appeared in images
    and total_freq_c is the total number of pixels of c in the total pixels of the images where c appeared.

    INPUTS:
    - image_files(list): a list of image_filenames which element can be read immediately
    - num_classes(int): the number of classes of pixels in all images

    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    #Initialize all the labels key with a list value
    label_to_frequency_dict = {}
    for i in xrange(num_classes):
        label_to_frequency_dict[i] = []

    for n in xrange(len(image_files)):
        image = imread(image_files[n])

        #For each image sum up the frequency of each label in that image and append to the dictionary if frequency is positive.
        for i in xrange(num_classes):
            class_mask = np.equal(image, i)
            class_mask = class_mask.astype(np.float32)
            class_frequency = np.sum(class_mask)

            if class_frequency != 0.0:
                label_to_frequency_dict[i].append(class_frequency)

    class_weights = []

    #Get the total pixels to calculate total_frequency later
    total_pixels = 0
    for frequencies in label_to_frequency_dict.values():
        total_pixels += sum(frequencies)

    for i, j in label_to_frequency_dict.items():
        j = sorted(j) #To obtain the median, we got to sort the frequencies

        median_frequency = np.median(j) / sum(j)
        total_frequency = sum(j) / total_pixels
        median_frequency_balanced = median_frequency / total_frequency
        class_weights.append(round(median_frequency_balanced, 3))

    #Set the last class_weight to 0.0 as it's the background class
    #class_weights[-1] = 0.0

    return class_weights

if __name__ == "__main__":
    print(median_frequency_balancing(image_files, num_classes=11))
    print(ENet_weighing(image_files, num_classes=11))