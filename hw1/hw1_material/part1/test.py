from ctypes import sizeof
import numpy as np
import cv2

# gaussian_images_ovtave1 = []

# # print("hi")

# a =np.array ([[1,2],[3,4],[5,6]])
# print(a)
# print(a/2)
image = cv2.imread('./testdata/1.png', 0).astype(np.float32)
# image2 = cv2.imread('./testdata/2.png', 0).astype(np.float32)/255

print(image.shape)

    


# for num_octave in  range(num_octaves):
#     for num_guassian_image in range(num_guassian_images_per_octave):
#         gaussian_images.append(cv2.GaussianBlur (image1,(3, 3),2**(1/4)**(num_guassian_image)))
#         print("append ! ",len(gaussian_images),"sigma",num_guassian_image)

#         if(num_guassian_image==num_guassian_images_per_octave-1) :
#             print("resize!")
#             image1=cv2.resize(gaussian_images[4], (image_shape[0]//2, image_shape[1]//2), interpolation=cv2.INTER_NEAREST)   

# dog_images = []
# for num_octave in  range(num_octaves):
#     for num_DoG_image in range(num_DoG_images_per_octave):
#         dog_images.append(cv2.subtract(gaussian_images[(num_DoG_image+num_octave*num_DoG_images_per_octave+1)], gaussian_images[(num_DoG_image+num_octave*num_DoG_images_per_octave)]))
#         print("append ! ",len(dog_images))


# print(image)
# print('-------------------')
# print(image2)
# cv2.imshow('img',image)
# cv2.imshow('img',image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
a=1
c=str(a)+'hi'
print(c)