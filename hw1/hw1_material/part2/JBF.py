import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s  #half of wndw
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)


### TODO ###
        
        padded_guidance = padded_guidance.astype('float64')
        padded_img = padded_img.astype('float64')

        # step 1: Spatial kernel(gaussian)
        Sp=np.zeros((self.wndw_size, self.wndw_size))

        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
               Sp[i][j]=np.exp((-1)*(np.square(i-self.pad_w)+np.square(j-self.pad_w))/2/np.square(self.sigma_s)) 

        output=np.zeros(img.shape)
        padded_guidance=np.divide(padded_guidance,255)
       

        # step 2: Start to count every pixel    
        for i in range(self.pad_w,padded_guidance.shape[0]-self.pad_w):
            for j in range(self.pad_w, padded_guidance.shape[1]-self.pad_w):

            ## step 2-1: construct Range kernel
                current_pixel = padded_guidance[i][j]
                Rg_kernel=padded_guidance[i-self.pad_w:i+self.pad_w+1,j-self.pad_w:j+self.pad_w+1]
                Rg=np.square(Rg_kernel-current_pixel)/((-2)*np.square(self.sigma_r))
                if(len(Rg.shape)==3):
                   Rg = Rg.sum(axis=2)  
                Rg=np.exp(Rg)

            ## step 2-2: construct weight ,G
                hs_hr=np.multiply(Rg,Sp)
                weight=(hs_hr.sum(axis=1)).sum(axis=0)


        # step 3: Do joint_bilateral_filter 
                wndw=padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]
                for k in range(img.shape[2]):
                   output[i-self.pad_w, j-self.pad_w, k] = np.multiply(hs_hr,wndw[:,:,k]).sum(axis=1).sum(axis=0)/weight
                    
        return np.clip(output, 0, 255).astype(np.uint8)