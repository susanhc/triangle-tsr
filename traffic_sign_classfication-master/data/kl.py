import os
import cv2

for name in os.listdir('./9/'):
    na=name.split('_')[0]
    image = cv2.imread('./9/'+name)
    print(na)
    cv2.imshow('iamge',image)
    cv2.waitKey(0)