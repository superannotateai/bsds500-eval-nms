import numpy as np
import os
from os.path import join, isdir
import torch
import torch.nn.functional as F
from . import suppress
import cv2
from tqdm import tqdm

def convTri(I, r):
    padded = np.pad(I, (r,r), mode='symmetric')
    f = np.arange(1,r+1)
    f = np.concatenate((f, np.array([r+1]), f[::-1]))
    f = f[np.newaxis,:]/(r+1)**2
    f = f.T.dot(f)
    img = torch.from_numpy(padded[np.newaxis,np.newaxis,:,:]).float()
    kernel = torch.from_numpy(f[np.newaxis,np.newaxis,:,:]).float()
    return F.conv2d(img,kernel).numpy()

def nms_all_edges(pred_path,ext):
    save_path = join(pred_path,'test')
    if not isdir(save_path):
        os.makedirs(save_path)
        
    for img_path in tqdm(os.listdir(pred_path)):
        if img_path.endswith(ext):
            E = cv2.imread(join(pred_path,img_path), 0)/255
            Oy, Ox = np.gradient(convTri(E,4).squeeze(), 1)
            _, Oxx = np.gradient(Ox, 1)
            Oyy, Oxy = np.gradient(Oy, 1)
            O=np.mod(np.arctan(np.divide(np.multiply(Oyy,np.sign(-Oxy)),Oxx+1e-5)),np.pi)
            edge = suppress.suppress(E.astype(np.float),O.astype(np.float),2,5,1.01,8)
            cv2.imwrite(join(save_path,img_path),255*edge)
