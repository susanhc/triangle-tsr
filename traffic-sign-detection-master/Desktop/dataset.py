import os
import shutil
import cv2


for filename in os.listdir('./train/'):
	print(filename[5])
	if filename[5]!='_' or filename[9]=='j':
		shutil.move('./train/%s'%(filename),'./train2/')
















'''
with open('train.txt') as f:
	lines=f.readlines()
	filename=[]
	for line in lines:
		name=line.split(';')[0]
		filename.append(name)
name=[]
for filename in os.listdir('./train/'):
	name.append(filename)
	na=sorted(name)

for a in filename:
	for b in na:
		if a==b:
			print('a=',a)
			img=cv2.imread('./train/%s'%(a))
			cv2.imshow('img',img)
			cv2.imwrite('./train2/',img)
			#shutil.move('./train/%s'%(a),'./train2/')
			cv2.waitKey(0)'''