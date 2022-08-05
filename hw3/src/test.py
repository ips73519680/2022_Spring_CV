from utils import solve_homography, warping
import numpy as np
import cv2


output3_1 = cv2.imread('./output3_1.png',cv2.IMREAD_GRAYSCALE)
output3_2 = cv2.imread('./output3_2.png', cv2.IMREAD_GRAYSCALE)

print(type(output3_1[0][0]))


different=cv2.subtract(output3_1, output3_2).astype('uint8')
different=255-different


imgs=np.hstack([output3_1,output3_2])


# for pixel in different:
#     if(pixel.all):
#         print('diff')
# cv2.imwrite('different_output3_1_output3_1.png', different)
# cv2.imshow('different_new',different)
# cv2.imshow('imgs',imgs)
# cv2.waitKey(0)
# cv2.destroyAllWindows()