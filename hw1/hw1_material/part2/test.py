import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


a=np.array([1,2,3])
b=np.array([[1,2,3],[4,5,6]])

print(a)
print(b[1,1])
print(b[1][1]-a)