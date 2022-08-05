import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping
import math

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])
    w_now=0 
    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):

        im1 = imgs[idx]
        im2 = imgs[idx + 1]
        w_now  += im1.shape[1] # add width  
        # TODO: 1.feature detection & matching
        kp1, des1 = orb.detectAndCompute(im1,None)
        kp2, des2 = orb.detectAndCompute(im2,None)
      
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        matches=matches[:100]
        # print(len(matches))
        # print(matches)
        u=[]
        v=[]
        for match in matches:
            u.append(kp1[match.queryIdx].pt)
            v.append(kp2[match.trainIdx].pt)

        u=np.array(u)
        v=np.array(v)      

        # TODO: 2. apply RANSAC to choose best H
        iters = 2000
        threshold  = 4
        best_H = np.eye(3)
        last_total_inlier=0
        
        for i in range(iters):
            random_int=[]
            for j in range(4):
                random_int.append(random.randint(0, len(u)-1))
            random_int=np.array(random_int)
            random_u=[u[int] for int in random_int]
            random_v=[v[int] for int in random_int]
            random_u=np.array(random_u)
            random_v=np.array(random_v)            
            H = solve_homography(random_v, random_u)  # img2 -H-> img1   


        # count inlier
            total_inlier = 0
            one = np.ones((1,len(u))) 
            M_img1=np.concatenate((np.transpose(v), one), axis=0) # homogeneous coordinate
            M_transformed  = np.dot(H,M_img1) # homogeneous coordinate
            M_transformed_Ordinary =np.divide(M_transformed[:-1], M_transformed[-1,:])  # Ordinary Coordinate

            errs=np.linalg.norm((M_transformed_Ordinary-np.transpose(u)),ord=1,axis=0)

            total_inlier=sum(errs<threshold)
            


            if (last_total_inlier<total_inlier):
                best_H=H
                last_total_inlier=total_inlier

        # TODO: 3. chain the homographies
        last_best_H = last_best_H.dot(best_H)

            # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0, im2.shape[0], w_now, w_now+im2.shape[1], direction='b') 

    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)