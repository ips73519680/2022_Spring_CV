from itertools import count
from cv2 import imshow, normalize
import numpy as np
import cv2


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
       
        image_shape=image.shape[0:2]
        for num_octave in  range(self.num_octaves):
            for num_guassian_image in range(self.num_guassian_images_per_octave):
                
                if(num_octave==0 and num_guassian_image==0):
                    gaussian_images.append(image)
                elif(num_octave==1 and num_guassian_image==0):
                    gaussian_images.append(resize_image)

                elif (num_octave==0):  
                    gaussian_images.append(cv2.GaussianBlur (image,(0,0),self.sigma**(num_guassian_image)))
                
                elif(num_octave==1) :
                    gaussian_images.append(cv2.GaussianBlur (resize_image,(0,0),self.sigma**(num_guassian_image)))

                if(num_guassian_image==4 and num_octave==0):
                     resize_image=cv2.resize(gaussian_images[4], (image_shape[1]//2, image_shape[0]//2), interpolation=cv2.INTER_NEAREST)
                    
                    

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for num_octave in  range(self.num_octaves):
            for num_DoG_image in range(self.num_DoG_images_per_octave):
                first=(num_DoG_image+num_octave*(self.num_guassian_images_per_octave)+1)
                second=(num_DoG_image+num_octave*(self.num_guassian_images_per_octave))
                dog_images.append(cv2.subtract(gaussian_images[second], gaussian_images[first]))

     
        # normalize_DoG=[]
        # for item in dog_images:
        #     max=np.max(item)
        #     min=np.min(item)
        #     normalize_DoG.append((np.array(item)-min)/(max-min)*255)

        # count=0
        # for item in normalize_DoG:
        #     name=str(count)+'new'+'.png'    
        #     cv2.imwrite(name, item)
        #     count+=1    

       
        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints=[]
        current_h=image_shape[0]
        current_w=image_shape[1]

        for num_octave in  range(self.num_octaves):
            if(num_octave==1):
                current_h=int(image_shape[0]//2)
                current_w=int(image_shape[1]//2)
            for num_DoG_image in range(1,self.num_DoG_images_per_octave-1):
                current_img=num_DoG_image+self.num_DoG_images_per_octave*num_octave
                last_img=num_DoG_image+self.num_DoG_images_per_octave*num_octave-1
                next_img=num_DoG_image+self.num_DoG_images_per_octave*num_octave+1
                
                for x in range(1,current_h-1):
                    for y in range(1,current_w-1):
                        current_point=dog_images[current_img][x][y]
                        
                        # check if keypoint
                        if(abs(current_point)<self.threshold):
                            continue
                        else:
                            
                            #for local max
                            max_check=1
                            for i in range(-1,2):
                                if(max_check==0):
                                    break
                                for j in range(-1,2): 
                                    if(current_point<dog_images[next_img][i+x][j+y] or current_point<dog_images[last_img][i+x][j+y] or current_point<dog_images[current_img][i+x][j+y]):
                                        max_check=0
                                        break
                            
                            if(max_check==1 and num_octave==0):
                                keypoints.append([x,y])                              
                                continue
                            if(max_check==1 and num_octave==1):
                                keypoints.append([x*2,y*2])
                                continue     

                            

                            # for local min
                            min_check=1
                            for i in range(-1,2):
                                if(min_check==0):
                                    break
                                for j in range(-1,2):
                                    if(current_point>dog_images[last_img][i+x][j+y] or current_point>dog_images[next_img][i+x][j+y] or  current_point>dog_images[current_img][i+x][j+y]):
                                        min_check=0
                                        break
                                   
                            if(num_octave==0 and min_check==1):
                                 keypoints.append([x,y])
                                 continue
                            if(num_octave==1 and min_check==1):
                                keypoints.append([x*2,y*2])
                                continue


        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints=np.unique(keypoints,axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 

        return keypoints
