import pycocotools
from ultralytics import YOLO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import yaml
import torch
import shutil
import wandb
from PIL import Image
import shutil

path_best_weights="../runs/detect/train11/weights/best.pt"
model = YOLO(path_best_weights)

names={0:"door",1:"cabinetDoor",2:"refrigeratorDoor",3:"window",4:"chair",5:"table",
    6:"cabinet",7:"couch",8:"openedDoor",9:"pole"}

COLORS=np.random.uniform(0,255,(10,3))

def clear_fodler():
    folder = 'archive/live_test/images'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def show_bbox(img_name,img_dir,label_dir):
    img_path=os.path.join(img_dir,img_name)
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h,w,_=img.shape
    img=img.copy()
    
    try:
        label_path=os.path.join(label_dir,img_name[:-4]+".txt")
        label=pd.read_csv(label_path,sep=" ",header=None).values
        classes=label[:,0]
        boxes=label[:,1:]
        
        for i,box in enumerate(boxes):
            cls_id=int(classes[i])
            if classes[i] in [0, 4, 5, 7]:
                text=names[cls_id]
                color=COLORS[cls_id]
                xmin=int((box[0]-box[2]/2)*w)
                ymin=int((box[1]-box[3]/2)*h)
                xmax=int((box[0]+box[2]/2)*w)
                ymax=int((box[1]+box[3]/2)*h)
                cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color,3)
                y=ymin-10 if ymin-10>10 else ymin+20
                cv2.putText(img,text,(xmin,y),cv2.FONT_HERSHEY_SIMPLEX,1.5,color,3)
    except:
        pass
    
    return img

def read_video(path):
  cap = cv2.VideoCapture(path)
  arr = []
  while cap.isOpened():
    success, image = cap.read()
    if not success: break
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    arr.append(image)
  cap.release()
  return arr

def save_frames(uploaded_file):
    if os.path.exists("archive/{}".format(uploaded_file.name)):
        arr = read_video("archive/{}".format(uploaded_file.name))
        print(arr[0].shape)

        if len(arr) > 0:
            for frameid, npimg in enumerate(arr):
                img = Image.fromarray(npimg, 'RGB')
                img.save("archive/live_test/images/sample{}.jpg".format(frameid), "JPEG")
    
        print(len(arr), arr[0].shape)
            
def run_prediction():
    test_imgs_dir = 'archive/live_test/images'

    with torch.no_grad():
        results=model.predict(source=test_imgs_dir,conf=0.3,iou=0.75)
        
    return results

def detect(uploaded_file):
    prediction_dir="predictions/prediction_yolo8vl"
    status = save_frames(uploaded_file)
    results = run_prediction()
    
    test_img_list=[]
            
    for result in results:
        if len(result.boxes.xyxy):
            name=result.path.split("/")[-1].split(".")[0]
            boxes=result.boxes.xywhn.cpu().numpy()
            classes=result.boxes.cls.cpu().numpy()
            
            test_img_list.append(name)
            
            label_file_path=os.path.join(prediction_dir,name+".txt")
            with open(label_file_path,"w+") as f:
                for cls_obj,box in zip(classes,boxes):
                    text=f"{int(cls_obj)} "+" ".join(box.astype(str))
                    f.write(text)
                    f.write("\n")
                    
    arr_pred = []

    for pred in range(len([i for i in os.listdir("archive/live_test/images") if not i.endswith(".txt")])):
        img = show_bbox("sample{}.jpg".format(pred),"archive/live_test/images","archive/live_test/images/")
        arr_pred.append(img)
        
    save_dir = "archive/live_test/sample_pred.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    height, width, layers = arr_pred[0].shape
    size = (width,height)
    fps = 30
    out = cv2.VideoWriter(save_dir, fourcc, fps, size)
    for img in arr_pred:
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(image)
    out.release()
    
    return True
