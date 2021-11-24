import os
import cv2 
import torch
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import torchvision.transforms as T


# Directory of video folder
video_root = '/ssd/viodata/dataset/train/'
output_root = '/home/lab/Videos/five_crop/train/'
# Make a list of file ending with .MP4
video_files = glob(video_root + '*.mp4')

fps = 24

for name in tqdm(video_files):

    cap = cv2.VideoCapture(name)
    frame = int(cap.get(7))

    # Initialize five videowriter
    out_name = name.replace(video_root, output_root)[:-4]

    output_videos = [cv2.VideoWriter(out_name + i + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (224, 224)) for i in ['__0','__1','__2','__3','__4']]
    
    for index in range(frame):
        _, img = cap.read()
        
        # Transform the OpenCV Img 2 PIL Img
        orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img=Image.fromarray(orig_img)

        # Apply five-crop with torch
        crop_imgs = T.FiveCrop(size=(224, 224))(pil_img)

        # Transform each PIL Img back 2 OpenCV Img
        for  vid, crop_img in zip(output_videos, crop_imgs):
            numpy_image=np.array(crop_img)  
            opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            vid.write(opencv_image)

    # Release all videos
    #map(lambda vid: vid.release(), output_videos)
    for v in output_videos:
        v.release()
    cap.release()

    # break
            

