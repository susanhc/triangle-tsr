import os
import shutil



for root,dirs,files in os.walk('./train/'):
	if dirs!=[]:
		dir1=dirs
		print(dir1[0])

for i in range(len(dir1)):
	b=dir1[i]
	for name in os.listdir('./proposal_train_affine'):
		a=name[0:2]
		if a[1]=='_':
			a=a[0]
		else:
			a=a
		if a==b:
			print('a=',a)
			shutil.move('./proposal_train_affine/'+name,'./train/%s/'%(b))
