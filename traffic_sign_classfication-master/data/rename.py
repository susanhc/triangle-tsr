import os

filename=os.listdir('./10')
for name in filename:
	a=name.split('_')
	cla=a[0]
	name=os.path.join('./10',name)
	rename=os.path.join('./10','{}'.format('10'+'_'+a[1]))
	os.rename(name,rename)
