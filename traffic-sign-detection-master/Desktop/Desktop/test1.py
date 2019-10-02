import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

annotations_path ='./Annotations/'

def write_xml(imgname,filepath,labeldicts):
  
    root = ET.Element('annotation')
    ET.SubElement(root,'folder').text='train'                             
   
    ET.SubElement(root, 'filename').text = str(imgname)
                      
    	
    for labeldict in labeldicts:
    	size=ET.SubElement(root,'size')
    	ET.SubElement(size,'width').text = labeldict['width']
    	ET.SubElement(size,'height').text = labeldict['height']
    	ET.SubElement(size,'depth').text = '3'

    	objects = ET.SubElement(root, 'object')                       
    	ET.SubElement(objects, 'name').text = labeldict['name']
    	ET.SubElement(objects, 'pose').text = 'unspecified'
    	ET.SubElement(objects, 'truncated').text= '0'
    	ET.SubElement(objects, 'difficult').text= '0'
    
    	bndbox = ET.SubElement(objects,'bndbox')
    	ET.SubElement(bndbox, 'xmin').text = labeldict['xmin']
    	ET.SubElement(bndbox, 'ymin').text = labeldict['ymin']
    	ET.SubElement(bndbox, 'xmax').text = labeldict['xmax']
    	ET.SubElement(bndbox, 'ymax').text = labeldict['ymax']
    
    tree = ET.ElementTree(root)
    tree.write(filepath, encoding='utf-8')	

def test(content): 

	image_id=content[0]
	width=content[1]
	height=content[2]
	xmin=content[3]
	ymin=content[4]
	xmax=content[5]
	ymax=content[6]
	name=content[7]

	new_dict={'width':width,
			'height':height,
			'xmin':xmin,
			'ymin':ymin,
			'xmax':xmax,
			'ymax':ymax,
			'name':name}
			
	img_ids.append(image_id)
	new_dicts.append(new_dict)
		
			
	return img_ids, new_dicts

					

if __name__=='__main__':

	with open('./train.txt') as f:
		contents=f.readlines()
		
		img_ids = []
		new_dicts = []

		for content in contents:
			content=content.split('\n')
			print(content)

			img_ids, new_dicts = test(content)
	

			#write_xml(img_ids,annotations_path+str(img_ids),new_dicts)
