import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/2_setting.txt', help='path to setting file')
    parser.add_argument('--sigma_s', default=1, type=int, help='sigma of spatial kernel')
    parser.add_argument('--sigma_r', default=0.05, type=float, help='sigma of range kernel')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # img_gray_cv2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gray1=np.dot(img_rgb[...,:3], [0.1,0.0,0.9]).astype(np.uint8)
    # img_gray2=np.dot(img_rgb[...,:3], [0.2,0.0,0.8]).astype(np.uint8)
    img_gray3=np.dot(img_rgb[...,:3], [0.2,0.8,0.0]).astype(np.uint8)
    # img_gray4=np.dot(img_rgb[...,:3], [0.4,0.0,0.6]).astype(np.uint8)
    # img_gray5=np.dot(img_rgb[...,:3], [1.0,0.0,0.0]).astype(np.uint8)



    # ### TODO ###
    JBF = Joint_bilateral_filter(args.sigma_s, args.sigma_r)
  
    # parameter： joint_bilateral_filter(self, img, guidance)
    # generate img :
   
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    # jbf_out_cv = JBF.joint_bilateral_filter(img_rgb, img_gray_cv2).astype(np.uint8)
    jbf_out_1 = JBF.joint_bilateral_filter(img_rgb, img_gray1).astype(np.uint8)
    # jbf_out_2 = JBF.joint_bilateral_filter(img_rgb, img_gray2).astype(np.uint8)
    jbf_out_3 = JBF.joint_bilateral_filter(img_rgb, img_gray3).astype(np.uint8)
    # jbf_out_4 = JBF.joint_bilateral_filter(img_rgb, img_gray4).astype(np.uint8)
    # jbf_out_5 = JBF.joint_bilateral_filter(img_rgb, img_gray5).astype(np.uint8)


    # # compute the cost :
    # jbf_error_cv = np.sum(np.abs(jbf_out_cv.astype('int32')-bf_out.astype('int32')))
    jbf_error_1 = np.sum(np.abs(jbf_out_1.astype('int32')-bf_out.astype('int32')))
    # jbf_error_2 = np.sum(np.abs(jbf_out_2.astype('int32')-bf_out.astype('int32')))
    jbf_error_3 = np.sum(np.abs(jbf_out_3.astype('int32')-bf_out.astype('int32')))
    # jbf_error_4 = np.sum(np.abs(jbf_out_4.astype('int32')-bf_out.astype('int32')))
    # jbf_error_5 = np.sum(np.abs(jbf_out_5.astype('int32')-bf_out.astype('int32')))


    # # show the cost :
    # print('jbf_error_cv:',jbf_error_cv)
    print('jbf_error_1:',jbf_error_1)
    # print('jbf_error_2:',jbf_error_2)
    print('jbf_error_3:',jbf_error_3)
    # print('jbf_error_4:',jbf_error_4)
    # print('jbf_error_5:',jbf_error_5)



    # # show the image
    # name_hg='2_jbf_out_h'+'_gray'+'.png'    
    # cv2.imwrite(name_hg, img_gray3)
    # name_h='2_jbf_out_h'+'_Filtered RGB'+'.png'    
    # cv2.imwrite(name_h,cv2.cvtColor(jbf_out_3,cv2.COLOR_RGB2BGR))
    # name_lg='2_jbf_out_l'+'_gray'+'.png'    
    # cv2.imwrite(name_lg, img_gray1)
    # name_l='2_jbf_out_l'+'_Filtered RGB'+'.png'    
    # cv2.imwrite(name_l, cv2.cvtColor(jbf_out_1,cv2.COLOR_RGB2BGR))


    # imgs = np.hstack([cv2.cvtColor(jbf_out_3,cv2.COLOR_RGB2BGR),cv2.cvtColor(jbf_out_1,cv2.COLOR_RGB2BGR)])
    # cv2.imshow('imgs',imgs)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()