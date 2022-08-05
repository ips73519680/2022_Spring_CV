from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
import cv2

def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
    #                                                                          #                                                               
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#                                                                                   
    # or equivalently the number of entries in each image's histogram.         #
    #                                                                          #
    # You will construct SIFT features here in the same way you did in         #
    # build_vocabulary (except for possibly changing the sampling rate)        #
    # and then assign each local feature to its nearest cluster center         #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''

    # print('get_bags_of_sifts start')

    image_feats=[]
    step_x=1
    step_y=1

    # load the vocabulary
    with open('vocab.pkl', 'rb') as handle:
        vocab = pickle.load(handle)

    # 
    for img_path in image_paths:
        img=cv2.imread(img_path)
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # this method takes the input arguments as Y-axis, then X-axis.
        frames, descriptors = dsift(img_gray, step=[step_y,step_x],window_size=4, fast=True)
        print('descriptors.shape',np.array(descriptors).shape)
        descriptors=descriptors[::5]
        print('descriptors.shape__[::5]',np.array(descriptors).shape)

        # histogram

        print('vocab',np.array(vocab).shape)
        # distance.cdist(x1,x2) :
        ## row 1 = distance of  each x2's element to x1's first element  第一行数据表示的是x1数组中第一个元素点与x2数组中
        dist=distance.cdist(vocab,descriptors)

        # np.argmin : check the numbers of feature which matched vocabs
        min=np.argmin(dist,axis=0)
        
        hist,bin_edges = np.histogram(min,bins=len(vocab))
        hist_norm=[float(i)/sum(hist) for i in hist]
        image_feats.append(hist_norm)

    image_feats=np.matrix(image_feats)
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats
