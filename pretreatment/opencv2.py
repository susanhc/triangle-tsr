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
    '''
    na=filename.strip('css (')
    nam=na.strip(').bmp')
    name=int(nam)
    print(name)
    '''
    name=filename
    #name=int(filename.strip('.png'))

    img=cv2.imread('./data/'+filename)
    
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
    #cv2.imshow('dilate',dilate)
    cv2.waitKey(0)

    contours, hierarchy=cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    c_true=[]
    for i in range(len(contours)):
        cnt=contours[i]
        #area=cv2.contourArea(cnt)
        x,y,w,h=cv2.boundingRect(cnt)
        a=w/h
        if w>15 and w<250 and h>15 and h<250:
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
    contours, hierarchy=cv2.findContours(binary, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for i in range(len(contours)):
            cnt=contours[i]
            con=[]
            con.append(cnt)
            #x,y,w,h=cv2.boundingRect(contours[i])
            #cv2.rectangle(img,(x,y),(x+w,y+h),(153,153,0),2)
            cv2.drawContours(img,con,-1,(255,255,255),thickness=-1)
            #cv2.imshow('green',img)
            #cv2.imwrite('./data/5.jpg',img)
            cv2.waitKey(0)

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
    contours, hierarchy=cv2.findContours(binary, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c_true=[]
    for i in range(len(contours)):
        cnt=contours[i]
        #area=cv2.contourArea(cnt)
        x,y,w,h=cv2.boundingRect(cnt)
        a=w/h
        if w>20 and w<200 and h>20 and h<200:
            c_true.append(cnt)

    return img,c_true
def canny(img):
    edge=cv2.Canny(img,100,300)
    contours, hierarchy=cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c_true=[]
    for cnt in contours:
        c_true.append(cnt)
        epsilon=0.01*cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,epsilon,True)
        hull=cv2.convexHull(cnt)
        #cv2.drawContours(img,[cnt],-1,(0,255,0),2)

    return img,c_true

def triangle_dete(c_true):
    shape=[]
    for i in range(len(c_true)):
        c_shape=c_true[i]
        hull = cv2.convexHull(c_shape)
        length=len(hull)
        for i in range(len(hull)):
            cv2.circle(img, tuple(hull[i][0]), 2, (0, 0, 255), 2, 8, 0)
            cv2.line(img,tuple(hull[i][0]),tuple(hull[(i+1)%length][0]),(0,255,0),2)
        #cv2.imshow('line',img)
        cv2.waitKey(0)
        
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
                minAngle = 30
                maxAngle = 75
                #how many degs can the side roughly parallel to the x axis be rotated in either way
                rotationError = 8
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
        xmin=[]
        xmin2=[]
        l=[]
        l3=[]
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
                #print('xmax1=',xmax1)
            xmin.append([xmin1,ymin1,xmax1,ymax1])
            #print([box]) 
            #计算矩形的行列 
            h1=max([box][0][0][1],[box][0][1][1],[box][0][2][1],[box][0][3][1]) 
            h2=min([box][0][0][1],[box][0][1][1],[box][0][2][1],[box][0][3][1]) 
            l1=max([box][0][0][0],[box][0][1][0],[box][0][2][0],[box][0][3][0]) 
            l2=min([box][0][0][0],[box][0][1][0],[box][0][2][0],[box][0][3][0])
            l.append([l2,h2,l1,h1])
            #加上防错处理，确保裁剪区域无异常 
        #xmin2=list(set(xmin))
        #xmin2.sort(key=xmin.index)
        e=l[0]
        l3.append(l[0])
        for j in range(len(l)):
            if l[j]!=e:
                l3.append(l[j])
                e=l[j]
        n=1
        o=[]
        f=open('./data_no_green.txt','a')
        f.write(name+' ')
        for k in range(len(l3)):
            coords=l3[k]
            xmin=coords[0]
            ymin=coords[1]
            xmax=coords[2]
            ymax=coords[3]
            #name=str(name)+'_'+str(n)+'.jpg'
            n+=1
            if xmax-xmin>0 and ymax-ymin>0:
                #裁剪矩形区域 
                temp=img[ymin:ymax,xmin:xmax]
            #显示裁剪后的标志
            #cv2.imshow('sign'+str(j),temp)
                j=j+1
                xmin=str(xmin)
                ymin=str(ymin)
                xmax=str(xmax)
                ymax=str(ymax)
                #cv2.imwrite('./cropped3/%s'%(name),temp)
            f.write('%s %s %s %s '%(xmin,ymin,xmax,ymax))
        f.write('\n')
        '''
            with open('gt duo2.txt','r') as fo:
                lines=fo.readlines()
                for line in lines:
                    na=line.split(' ')[0]
                    clas=line.split(' ')[1]
                    #if na==name:
                        #f.writelines(';'+clas)
                        #f.writelines('\n')'''
    #cv2.imshow('img',img)
    #cv2.imwrite('./data4/6.jpg',img)
    cv2.waitKey(0)
def dele_photo(i):
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


if __name__ == "__main__":
    for filename in os.listdir('./data/'):
        name,img=colorthreshold(filename)
        #img=greenthreshold(img)
        img,c_true=yellowthreshold(img)
        #img,c_true=canny(img)
        shape=triangle_dete(c_true)
        crop_image(shape,name)
