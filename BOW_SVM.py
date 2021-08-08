# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 18:50:31 2021

@author: Malik Usama
"""

from PIL import Image
import numpy as np
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
import pickle
from scipy.spatial import distance
import os
from glob import glob
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



## Splitting data  yahan per  num_train_per_cat k elawa ek aur parameter test ka
def get_image_paths(base_dir, num_train_per_cat,num_test_per_cat):
    categories = os.listdir(base_dir)
    train_image_paths = []
    test_image_paths = []

    train_labels = []
    test_labels = []

    for category in categories:

        image_paths = glob(os.path.join(base_dir, category, '*.jpg'))
        for i in range(num_train_per_cat):
            train_image_paths.append(image_paths[i])
            train_labels.append(category)

        image_paths = glob(os.path.join(base_dir, category, '*.jpg'))
        for i in range(num_train_per_cat,num_train_per_cat+num_test_per_cat):
            test_image_paths.append(image_paths[i])
            test_labels.append(category)

    return train_image_paths, test_image_paths, train_labels, test_labels




## Dense SIFT
def D_SIFT(image_paths):
    bag_of_features = []
    for path in image_paths:
        img = np.asarray(Image.open(path),dtype='float32')
        frames, descriptors = dsift(img, step=8,size=[16,16], fast=True)
        bag_of_features.append(descriptors)
    bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
    return bag_of_features


# built dctionary
def visual_vocab(bag_of_features,vocab_size):    
    vocab = kmeans(bag_of_features, vocab_size, initialization="PLUSPLUS")        
    return vocab


## BOW from Vocab
def get_bags_of_word_rep(image_paths):
     with open('vocab_train_200.pkl', 'rb') as handle:
        vocab = pickle.load(handle)
     image_feats = []
     for path in image_paths:
        img = np.asarray(Image.open(path),dtype='float32')
        frames, descriptors = dsift(img, step=8,size=[16,16], fast=True)
        dist = distance.cdist(vocab, descriptors, metric='euclidean')
        idx = np.argmin(dist, axis=0)
        hist, bin_edges = np.histogram(idx, bins=len(vocab))
        hist_norm = [float(i)/sum(hist) for i in hist]
        
        image_feats.append(hist_norm)       
     image_feats = np.asarray(image_feats)
     return image_feats

## SVM

def svm_classify(train_image_feats, train_labels, test_image_feats):
    SVM = make_pipeline(StandardScaler(),SVC(C=1.0, kernel='rbf', degree=3,
                    gamma='scale', coef0=0.0, shrinking=True,
                    probability=False, tol=0.001, cache_size=200,
                    class_weight=None, verbose=False, max_iter=- 1, 
                    decision_function_shape='ovr', break_ties=False,
                    random_state=None))
    SVM.fit(train_image_feats, train_labels)
    pred_label = SVM.predict(test_image_feats)
    return pred_label

#visual
def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.rainbow):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#
def build_confusion_mtx(test_labels, predicted_categories):
    category=['BR', 'Sub', 'Ind', 'Ktc', 'LR', 'Cst', 'Fst',
                   'HWY', 'CT', 'MT', 'OC', 'ST', 'TB', 'Off', 'Str']
    matrix=confusion_matrix(test_labels,predicted_categories)
    plot_confusion_matrix(matrix,category)
    plt.show()
    return matrix
    
    
base_dir="C:/EME/8th sem/CV/assignment/#3/scenes/"
train_image_paths, test_image_paths, train_labels,test_labels=get_image_paths(base_dir,100,20)
b_o_features=D_SIFT(train_image_paths)
vocab_size=50
vocab_train=visual_vocab(b_o_features, vocab_size)
with open('vocab_train_200.pkl', 'wb') as handle:
    pickle.dump(vocab_train, handle)
bag_of_words_train=get_bags_of_word_rep(train_image_paths)
bag_of_words_test= get_bags_of_word_rep(test_image_paths);
pred_categ_res = svm_classify(bag_of_words_train, train_labels, bag_of_words_test)

cf_matrix=build_confusion_mtx(test_labels, pred_categ_res)


print(cf_matrix)