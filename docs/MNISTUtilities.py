# -*- coding: utf-8 -*-
import json
import gzip as gz
import codecs as cdc
import matplotlib.pyplot as plt
import numpy as np

# ### 2. MNIST Utilities

def to_int(x):
    return int(cdc.encode(x, 'hex'), 16)

def read_idx(file_path):
    with gz.open(file_path, 'rb') as f:
        mg = to_int(f.read(4))     #magic number determining type
        m = to_int(f.read(4))      #no. of samples
        nr, nc, shape = 0, 0, 0
        if (mg == 2051): #if image file
            nr = to_int(f.read(4))   #no of rows
            nc = to_int(f.read(4))   #no of columns
            shape = (m, nr, nc)
        elif (mg == 2049): #if label file
            shape = -1
        else:
            print('ERROR!: Unrecognizable file type.')
            return
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = np.reshape(data, shape)
        print(mg, m, nr, nc)
        return data

#Iterating over files
#Reading the data into the arrays for processing
def get_train_test_from_files(DIR_PATH, data_files):
    X_train = X_test = y_train = y_test = None
    for fn in data_files:
        filepath = DIR_PATH + fn
        print("Reading:", fn, " ...")
        if (fn.find('train') > -1):
            if (fn.find('images') > -1):
                X_train = read_idx(filepath)
                print(X_train.shape)
                #print('X_train')
            elif (fn.find('labels') > -1):
                y_train = read_idx(filepath)
                print(y_train.shape)
                #print('y_train')
        elif (fn.find('test') > -1):
            if (fn.find('images') > -1):
                X_test = read_idx(filepath)
                print(X_test.shape)
                #print('X_test')
            elif (fn.find('labels') > -1):
                y_test = read_idx(filepath)
                print(y_test.shape)
                #print('y_test')
    return X_train, X_test, y_train, y_test

#Sanity Check read data
def sanity_check(X_train, X_test, y_train, y_test, class_labels):
    r = 1
    c = 3
       
    print("Train Set:")
    m = X_train.shape[0]
    data = X_train
    label = y_train
    fig = plt.figure(figsize=(9, 3))
    for i in range(1, 4):
        k = np.random.randint(m)
        fig.add_subplot(r, c, i)
        plt.imshow(data[k].reshape((28, 28)), cmap='gray')
        plt.title(class_labels[label[k]], fontsize=16)
        plt.axis('off')
    plt.show()
    
    print("Test Set:")
    m = X_test.shape[0]
    data = X_test
    label = y_test
    fig = plt.figure(figsize=(9, 3))
    for i in range(1, 4):
        k = np.random.randint(m)
        fig.add_subplot(r, c, i)
        plt.imshow(data[k].reshape((28, 28)), cmap='gray')
        plt.title(class_labels[label[k]], fontsize=16)
        plt.axis('off')
    plt.show()

#Uses all the above functions to generate training and testing sets
def read_and_transform_MNIST_data(dir_path='data/mnist_data',category='digit-data', transpose=True, scale_features=True, flatten=True):
    #(1) Initialize the required paths and constants
    CATEGORY = category
    FILELIST_FN = 'mnist-db-filelist.json'
    DIR_PATH = dir_path
    #(2) Since we will be processing the 'letter-data', 
    #    lets get the files corresponding to the CATEGORY, letter data.
    #    The list of files are stored in 'mnist-db-filelist.json'
    data_files = []
    with open(DIR_PATH + FILELIST_FN, 'r') as jf:
        data_files = json.load(jf)[CATEGORY]

    #(3) Get the images and labels from the data files
    X_train, X_test, y_train, y_test = get_train_test_from_files(DIR_PATH, data_files)
    
    #(4) Transpose each image in the dataset
    if(transpose == True):
        X_train = np.transpose(X_train, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))
    
    #(5) Converting image matrices to image vectors
    if(flatten == True):
        m, nr, nc = X_train.shape
        X_train = np.reshape(X_train, (m, nr * nc))
        print(X_train.shape)
        m, nr, nc = X_test.shape
        X_test = np.reshape(X_test, (m, nr * nc))
    
    #(6) Feature scaling
    if(scale_features == True):
        X_train = X_train/255.0
        X_test = X_test/255.0
        
    return X_train, X_test, y_train, y_test


