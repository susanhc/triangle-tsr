#导包
import argparse
import cv2
import numpy as np
from imutils import paths
import imutils
from keras.models import load_model
from keras.preprocessing.image import img_to_array

NORM_SIZE = 32

def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to trained model model")
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-s", "--show", required=True, action="store_true", help="show predict image", default=False)
    args = vars(ap.parse_args())
    return args

def predict(args):
    print("loading model...")
    model = load_model(args['model'])

    print("loading image...")
    imagepath = list(paths.list_images(args['image']))
    for ip in imagepath:
        image = cv2.imread(ip)
        name=ip.split('/')[-1]
        orig = image.copy()

        #预处理
        image = cv2.resize(image,(NORM_SIZE, NORM_SIZE))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        #预测
        result = model.predict(image)[0]
        proba = np.max(result)
        label = str(np.where(result==proba)[0])
        print('label=',label[1])
        label1 = "{}: {:.2f}%".format(label, proba*100)
        with open('image_class.txt','a') as f:
            f.writelines(name+' ')
            f.writelines(label[1])
            f.writelines('\n')
        '''
        o=[]
        t=[]
        p=[]
        with open('gt_pre.txt') as f:
            lines=f.readlines()
            for line in lines:
                o.append(line)
                for k in range(len(o)):
                    str1=o[k]
                    s=str1.split('\'')[0]
                    t.append(s)
                    for i in range(0,5):
                        a=list(t[-1])
                        a.pop(2)
                        r=t[i]
                        p.append(r)
                p.append(label[1])
                p.append('\n')
                print('str1',p)
                print('########')
                print('o1=',o)
                o.insert(-1,str(label[1]))
                o.insert(-2,';')
                print('o2=',o)
                print('*******')
                '''
        if args['show']:
            output = imutils.resize(orig, width=400)
            cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Output", output)
            cv2.waitKey(0)


if __name__=="__main__":
    args = args_parse()
predict(args)