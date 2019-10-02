import os
import xml.etree.ElementTree as ET

classes_num = {"straight":0, "left":1, "right":2, "stop":3, "nohonk":4, "crosswalk":5,"background":6}
DATA_PATH='./data2/test_images/'
xml_dir='./data2/test_images/Annotations/'

with open('1.txt','a') as f:
	for xml_file in os.listdir(xml_dir):
		img_path=os.path.join(xml_dir,xml_file)
		tree = ET.parse(img_path)
		root = tree.getroot()
		image_path = ''
		labels = []

		for item in root:
			if item.tag == 'filename':
				name=item.text
				f.write(name+' ')
			elif item.tag == 'object':
				obj_name = item[0].text
				obj_num = str(classes_num[obj_name])
				xmin = item[4][0].text
				ymin = item[4][1].text
				xmax = item[4][2].text
				ymax = item[4][3].text
				labels.append([obj_name,xmin, ymin, xmax, ymax, obj_num])

				f.write(xmin+' '+ymin+' '+xmax+' '+ymax+' '+obj_num+' ')
		f.write('\n')
	#contents=f.readlines()
	#for content in contents:















































































































































































































