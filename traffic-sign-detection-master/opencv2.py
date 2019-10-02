import cv2
import numpy as np
import os
import math

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
       return v
    return v/norm
def angle(a, b, c):
    return math.degrees(math.acos(np.dot(
        (normalize(np.subtract(a, b))),
        (normalize(np.subtract(c, b))))))

def colorthreshold(filename):
    na=filename.strip('.png')
    name=int(na)
    print(name)

    img=cv2.imread('/home/hc/flush/mser/data3/'+filename)
    
    #cv2.imshow('image',img)

    #img=cv2.resize(image,(400,300),interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('img',img)     

    #转换为HSV空间
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #cv2.imshow('hsv',hsv)

    #设置HSV阈值到蓝色范围
    blue_lower=np.array([0,70,70])
    blue_upper=np.array([100,255,255])
    mask=cv2.inRange(hsv,blue_lower,blue_upper)
    #print('mask',type(mask),mask.shape)
    #cv2.imshow('mask',mask)

    blurred=cv2.blur(mask,(9,9))
    #cv2.imshow('blurred',blurred)
 
    #二值化
    ret,binary=cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
    #cv2.imshow('blurred binary',binary)

    #使区域闭合无空隙
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(21,7))
    closed=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
    #cv2.imshow('closed',closed)

    #腐蚀和膨胀
    erode=cv2.erode(closed,None,iterations=4)
    dilate=cv2.dilate(erode,None,iterations=4)
    cv2.imshow('dilate',dilate)

    image,contours, hierarchy=cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    c_true=[]
    for i in range(len(contours)):
        cnt=contours[i]
        #area=cv2.contourArea(cnt)
        x,y,w,h=cv2.boundingRect(cnt)
        a=w/h
        if w>5 and w<250 and h>5 and h<250:
            c_true.append(cnt)
            #cv2.rectangle(res,(x,y),(x+w,y+h),(153,153,0),2)     
        else:
            c_false=[]
            c_false.append(cnt)
            cv2.drawContours(img,c_false,-1,(0,0,0),thickness=-1)
    return name,img
def greenthreshold(img):
    #image=cv2.imread('/home/hc/flush/mser/cropped/'+photo2)

    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #cv2.imshow('hsv',hsv)

    #设置HSV阈值到蓝色范围
    green_lower=np.array([35,43,46])
    green_upper=np.array([77,255,255])
    mask=cv2.inRange(hsv,green_lower,green_upper)
    #print('mask',type(mask),mask.shape)
    #cv2.imshow('mask',mask)

    blurred=cv2.blur(mask,(9,9))
    #cv2.imshow('blurred',blurred)
 
    #二值化
    ret,binary=cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
    #cv2.imshow('blurred binary',binary)
    '''
    #使区域闭合无空隙
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(21,7))
    closed=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
    #cv2.imshow('closed',closed)

    #腐蚀和膨胀
    erode=cv2.erode(closed,None,iterations=4)
    dilate=cv2.dilate(erode,None,iterations=4)
    '''
    image,contours, hierarchy=cv2.findContours(binary, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for i in range(len(contours)):
            cnt=contours[i]
            con=[]
            con.append(cnt)
            #x,y,w,h=cv2.boundingRect(contours[i])
            #cv2.rectangle(img,(x,y),(x+w,y+h),(153,153,0),2)
            cv2.drawContours(img,con,-1,(255,255,255),thickness=-1)

    return img
def yellowthreshold(img):
    #image=cv2.imread('/home/hc/flush/mser/cropped/'+photo2)

    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #cv2.imshow('hsv',hsv)

    #设置HSV阈值到蓝色范围
    green_lower=np.array([0,70,70])
    green_upper=np.array([100,255,255])
    mask=cv2.inRange(hsv,green_lower,green_upper)
    #print('mask',type(mask),mask.shape)
    #cv2.imshow('mask',mask)

    blurred=cv2.blur(mask,(9,9))
    #cv2.imshow('blurred',blurred)
 
    #二值化
    ret,binary=cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
    #cv2.imshow('blurred binary',binary)
    '''
    #使区域闭合无空隙
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(21,7))
    closed=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
    #cv2.imshow('closed',closed)

    #腐蚀和膨胀
    erode=cv2.erode(closed,None,iterations=4)
    dilate=cv2.dilate(erode,None,iterations=4)
    '''
    image,contours, hierarchy=cv2.findContours(binary, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c_true=[]
    for i in range(len(contours)):
        cnt=contours[i]
        #area=cv2.contourArea(cnt)
        x,y,w,h=cv2.boundingRect(cnt)
        a=w/h
        if w>10 and w<200 and h>10 and h<200:
            c_true.append(cnt)

    return img,c_true

def triangle_dete(c_true):
    shape=[]
    for i in range(len(c_true)):
        c_shape=c_true[i]
        hull = cv2.convexHull(c_shape)
        hullLen = cv2.arcLength(hull, True)
        for i in np.arange(0.06, 2, 0.02):
            epsilon = i * hullLen
            approx = cv2.approxPolyDP(hull, epsilon, True)
            l = len(approx)
            if l < 3:
                #No good results :(
                break
            elif l == 3:
                #We have found a triangle approximation for this contour
                #See if all the contour angles are within our limits
                #and one triangle side is roughly parallel to the x axis
                minAngle = 10
                maxAngle = 85
                #how many degs can the side roughly parallel to the x axis be rotated in either way
                rotationError = 12
                anglesOk = True
                oneParallelSide = False

                for i in range(0, l):
                    p1 = tuple(approx[i][0])
                    p2 = approx[(i + 1) % l][0]
                    p3 = approx[(i + 2) % l][0]

                    ang = angle(p1, p2, p3)

                    if ang < minAngle or ang > maxAngle:
                        anglesOk = False
                        break

                    slopeAng = math.degrees(math.atan2(p2[0] * 1.0 - p1[0], p2[1] * 1.0 - p1[1]))
                    #Normalize to [0, 180)
                    if slopeAng < 0:
                        slopeAng += 180

                    if(slopeAng > 90 - rotationError and slopeAng < 90 + rotationError):
                        #The line is roughly parallel to the x axis
                        if oneParallelSide is False:
                            oneParallelSide = True
                        else:
                            #One other side was already found to be roughly parallel to
                            #the x axis, abort
                            anglesOk = False
                            oneParallelSide = False
                            break

                #if all angles were within limits, draw the triangle
                if anglesOk and oneParallelSide:
                    shape.append(c_shape)
    return shape

def crop_image(shape,name):
    b=len(shape)
    if b==0:
    	print('name=',name)
    if b!=0:
        j=0
        for con in shape:
            #轮廓转换为矩形 
            rect=cv2.minAreaRect(con) 
            #矩形转换为box 
            box=np.int0(cv2.boxPoints(rect)) 
            #在原图画出目标区域 
            cv2.drawContours(img,[box],-1,(0,0,255),2)
            for i in range(len(box)):
                xmin1=box[0][0]
                ymin1=box[0][1]
                xmax1=box[3][0]
                ymax1=box[3][1]
                #print('box=',box)
                #print('xmin1=',xmin1)
                #print('xmax1=',xmax1)
                f=open('./gt_prediction.txt','a')
                f.writelines('%s;%d;%d;%d;%d'%(filename,xmin1,ymin1,xmax1,ymax1)+'\n')

            #print([box]) 
            #计算矩形的行列 
            h1=max([box][0][0][1],[box][0][1][1],[box][0][2][1],[box][0][3][1]) 
            h2=min([box][0][0][1],[box][0][1][1],[box][0][2][1],[box][0][3][1]) 
            l1=max([box][0][0][0],[box][0][1][0],[box][0][2][0],[box][0][3][0]) 
            l2=min([box][0][0][0],[box][0][1][0],[box][0][2][0],[box][0][3][0])     
            #加上防错处理，确保裁剪区域无异常 
            if h1-h2>0 and l1-l2>0:
                #裁剪矩形区域 
                temp=img[h2:h1,l2:l1]
            #显示裁剪后的标志
            #cv2.imshow('sign'+str(j),temp)
                j=j+1
            #cv2.imwrite('./cropped/%d0%d.jpg'%(name,j),temp)
    cv2.imshow('img',img)
    cv2.waitKey(0)
def dele_photo(photo,na=[],name=[]):
    img=cv2.imread('./cropped/'+photo)
    file=int(photo.strip('.png'))
    na.append(file)
    name=sorted(na)

    for i in range(len(name)-1):
        img1=cv2.imread('./cropped/%d.jpg'%(name[i]))
        img2=cv2.imread('./cropped/%d.jpg'%(name[i+1]))
        height1=img1.shape[0]
        width1=img1.shape[1]

        height2=img2.shape[0]
        width2=img2.shape[1]
        if height1==height2 and width1==width2:
            im=img1.size
            c=0
            for a in range(height1):
                for b in range(width1):
                    if img1[a,b][0]==img2[a,b][0] and img1[a,b][1]==img2[a,b][1] and img1[a,b][2]==img2[a,b][2]:
                        c+=1
            if c*3==im: 
                os.remove('./cropped/%d.jpg'%(name[i]))
def object_detection_dir(test_dir, write_dir, write_bin_dir, model_path, result_txt):
    """test the images in test direction.
    Args:
        test_dir: test directory.
        write_dir: write directory of detection results.
        result_txt: a txt file used to save　all the detection results.
    Return: 
        none
    """
    img_names = os.listdir(test_dir)
    img_names = [img_name for img_name in img_names if img_name.split(".")[-1] == "jpg"]
    clf = joblib.load(model_path)
    if os.path.exists(result_txt):
        os.remove(result_txt)
    f = open(result_txt, "a")
    for index, img_name in enumerate(img_names):
        if index%50 == 0:
            print("total test image number = ", len(img_names), "current image number = ", index)
        
        row_data = os.path.join(test_dir, img_name) + " "
        save_path = os.path.join(write_dir,img_name.split(".")[0]+"_result.jpg")
        save_bin_path = os.path.join(write_bin_dir,img_name.split(".")[0]+"_bin.jpg")
        img = cv2.imread(os.path.join(test_dir, img_name))
        img_bin = preprocess_img(img, False)
        cv2.imwrite(save_bin_path, img_bin)

        rects = contour_detect(img_bin, min_area=500)
        img_bbx = img.copy()
        rows, cols, _ = img.shape
        for rect in rects:
            xc = int(rect[0] + rect[2]/2)
            yc = int(rect[1] + rect[3]/2)

            size = max(rect[2], rect[3])
            x1 = max(0, int(xc-size/2))
            y1 = max(0, int(yc-size/2))
            x2 = min(cols, int(xc+size/2))
            y2 = min(rows, int(yc+size/2))
            proposal = img[y1:y2, x1:x2]
            cls_prop = hog_extra_and_svm_class(proposal, clf)
            cls_prop = np.round(cls_prop, 2)
            cls_num = np.argmax(cls_prop)
            cls_name = cls_names[cls_num]
            
            if cls_name is not "background":
                row_data += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(cls_num) + " "
                cv2.rectangle(img_bbx,(rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0,0,255), 2)
                cv2.putText(img_bbx, cls_name+str(np.max(cls_prop)), (rect[0], rect[1]), 1, 1.5, (0,0,255),2)

        cv2.imwrite(save_path, img_bbx)
        f.write(row_data+"\n")

    f.close()

if __name__ == "__main__":
	for filename in os.listdir('./data3/'):
		name,img=colorthreshold(filename)
		img=greenthreshold(img)
		img,c_true=yellowthreshold(img)
		shape=triangle_dete(c_true)
		crop_image(shape,name)
		#for photo in os.listdir('./cropped4/'):
			#dele_photo(photo,na=[],name=[])
		'''
			for photo2 in os.listdir('./cropped/'):
				image=greenthreshold(photo2)
				deletegreen(image)'''
		
if __name__ == "__main__":
    test_dir = "/home/hc/flush/traffic-sign-detection-master/data/test_images/JPEGImages"
    write_dir = "/home/hc/flush/traffic-sign-detection-master/data/test_results"
    write_bin_dir = "/home/hc/flush/traffic-sign-detection-master/data/test_results_bin"
    result_txt = "/home/hc/flush/traffic-sign-detection-master/data/test_result.txt"
    object_detection_dir(test_dir, write_dir, write_bin_dir, "/home/hc/flush/traffic-sign-detection-master/svm_hog_classification/svm_model.pkl", result_txt)
    print("finished.")




