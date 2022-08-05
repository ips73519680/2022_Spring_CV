from PIL import Image
import numpy as np
import cv2

def get_tiny_images(image_paths):
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    '''
    Input : 
        image_paths: a list(N) of string where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''

    # print('get_tiny_images start')


    tiny_images=[]
    for img_path in image_paths:
        img=cv2.imread(img_path)
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        tiny_img=cv2.resize(img_gray,(16,16),interpolation=cv2.INTER_AREA)
        tiny_img_flatten=tiny_img.flatten()
        tiny_images.append(tiny_img_flatten)

    tiny_images = np.matrix(tiny_images)     


    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images
