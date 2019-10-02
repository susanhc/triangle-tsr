import os

path='./train/'
outpath='./train/'

index=484
for img in os.listdir(path):
	name=os.path.splitext(img)
	img_segment=str(name[0])
	org_name=os.path.join(path,img)
	changed_name=outpath+'%04d'%(index)+'.png'
	os.rename(org_name,changed_name)
	index+=1

