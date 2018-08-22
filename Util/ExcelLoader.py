# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 13:54:53 2018

@author: natsn
"""

import pandas as pd
import numpy as np
import WebScraper as WS
import urllib
import bs4
import pickle

fer_2013_filepath = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\DataSets\\fer2013\\fer2013.csv"
CAAD_filepath = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\DataSets\\CAAD\\dev_dataset.csv"
CAAD_data_dir = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\DataSets\\CAAD\\images\\"
ImageNet_dir = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\DataSets\\imagenet_fall11_urls\\"
Mnist_dir = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\DataSets\\mnist\\mnist_train.csv"
Website = "https://www.uwaterloo.ca"
cifar10_dir = "D:\\Desktop\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\DataSets\\cifar10"
cifar100_dir = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\DataSets\\cifar100"


# Returns the 36000+ Images of the Fer2013 Facial Recognition Dataset (10 Classes of facial expressions)
def get_fer2013_imgs(filepath = fer_2013_filepath):
    image_label_col = "emotion"
    image_data_col = "pixels"
    img_width = 48
    img_height = 48
    
    # Open file excel files data with pandas
    data = pd.read_csv(filepath)
    imgs_raw = data[image_data_col]
    labels = np.array(data[image_label_col])
    
    # Format raw images
    img_raw = np.array([str(imgs_raw[i]).split(' ') for i in range(imgs_raw.shape[0])], dtype = np.int)
    
    # Reshape images
    images = img_raw.reshape(img_raw.shape[0],img_width, img_height)
    return images, labels # (X&Y)

def get_cifar10(filepath = cifar10_dir, data_set = 1):
    #Image Properties
    CIFAR10_IM_WIDTH = 32
    CIFAR10_IM_HEIGHT = 32
    CIFAR10_IM_CHANNELS = 3
    fp = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Datasets\\cifar10\\data_batch_" + str(data_set) + ".txt"
    with open(fp, 'rb') as fo:
        datadict = pickle.load(fo, encoding='bytes')
    X = datadict[b'data']
    X = X.reshape(-1, CIFAR10_IM_WIDTH, CIFAR10_IM_HEIGHT, CIFAR10_IM_CHANNELS)
    y = datadict[b'labels']
    y = np.reshape(y,(len(y),1))
    return (X, y)
    
def get_cifar100(filepath = cifar100_dir, data_set = 1):
    CIFAR10_IM_WIDTH = 32
    CIFAR10_IM_HEIGHT = 32
    CIFAR10_IM_CHANNELS = 3
    fp = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Datasets\\cifar100\\train.txt"
    with open(fp, 'rb') as fo:
        datadict = pickle.load(fo, encoding='bytes')
    X = datadict[b'data']
    X = X.reshape(CIFAR10_IM_WIDTH, CIFAR10_IM_HEIGHT, CIFAR10_IM_CHANNELS, -1)
    y = datadict[b'labels']
    y = np.reshape(y,(len(y),1))
    return (X, y)

def get_mnist(filepath = Mnist_dir, train_size = 55000, norm = False):
    data = pd.read_csv(filepath, sep = ' ', header = None)
    data = data.values
    datums = [data[i][0] for i in range(len(data))]
    datums_segs = [datums[i].split(',') for i in range(len(data))]
    labels = np.array([datums_segs[i][0] for i in range(len(data))])
    imgs = np.array([datums_segs[i][1:] for i in range(len(data))], dtype = np.float)
    
    if imgs.shape[0] == labels.shape[0]:
        # Images are returns as NxHxW
        # Mnist Pixels are 28 by 28
        imgs = imgs.reshape(-1,28,28,1) # reshape
        # Return the training set, and the test set as (X_train, y_train, X_test, y_test)
        if norm:
            imgs /= 255 # Normalize pixel values
            return [imgs[0:train_size,:,:,:],labels[0:train_size], imgs[train_size:,:,:,:],labels[train_size:]]
        else:
            return [imgs[0:train_size,:,:,:],labels[0:train_size], imgs[train_size:,:,:,:],labels[train_size:]]
    else:
        return (-1, -1)
    
# Returns all the web data to the image folder
def get_CAAD_training_images(data_path = CAAD_filepath, img_dir_path = CAAD_data_dir):
    # Load in the excel file with pandas
    data = pd.read_csv(data_path)
    image_links = data["OriginalLandingURL"]
    i = 0
    WS.download_images(image_links[0])
    print('Downloaded Image ', i + 1)
    for i in range(10):
        WS.download_images(image_links[i])
        print('Downloaded Image ', i + 1)
        i = i + 1

def download_batch_imageNet(data_file = ImageNet_dir, batchnum = 1):
    #data = pd.read_csv(data_file, sep = " ", header = None)
    #webnet_base_url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='
    data_file = data_file + "batch_" + str(batchnum) + ".csv"
    (img_urls, img_ids) = get_imageNet_urls(data_file)
    img_dir = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\DataSets\\imagenet_fall11_urls\\images"
    # For each image id, pull the image from the website and store the value
    #for img_id,img_url in zip(img_ids,img_urls):
    WS.download_images(img_urls,img_dir,img_ids)
    
def get_imageNet_urls(data_file = ImageNet_dir):
    data = pd.read_table(data_file,sep = ' ', header = None)
    data = np.array(data[0])
    data = [data[i].split('\t') for i in range(len(data))]
    img_ids = np.array([data[i][0] for i in range(len(data))], dtype = np.str)
    img_urls = np.array([data[i][1] for i in range(len(data))], dtype = np.str)
    return (img_urls, img_ids)
    

















