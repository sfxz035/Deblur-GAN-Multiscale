import numpy as np
import os
import cv2 as cv
import random
import scipy.misc
import time
batch_index = 0

### blur dataset
def get_blur_files(data_dir,crop_size):
    imgs_coord = []
    list_dir = os.listdir(data_dir)
    for each in list_dir:
        dir = os.path.join(data_dir,each)
        inpt_lab = os.listdir(dir)
        inpt_dir = os.path.join(dir,inpt_lab[0])
        label_dir = os.path.join(dir,inpt_lab[2])
        img_files = os.listdir(inpt_dir)

        for file in img_files:
            inpt_path = os.path.join(inpt_dir,file)
            label_path = os.path.join(label_dir,file)
            img = cv.imread(inpt_path)
            x,y,z = img.shape
            coords_x = x // crop_size
            coords_y = y // crop_size
            coords = [(q, r) for q in range(coords_x) for r in range(coords_y)]
            for coord in coords:
                imgs_coord.append((inpt_path,label_path,coord))
    random.shuffle(imgs_coord)
    nub = str(len(imgs_coord))
    print('data number is '+nub)
    return imgs_coord
def get_inpt_label(imgtuple,size):
    img_inpt = cv.imread(imgtuple[0])
    img_label = cv.imread(imgtuple[1])

    x,y = imgtuple[2]
    img_inpt = img_inpt[x*size:(x+1)*size,y*size:(y+1)*size]
    img_label = img_label[x*size:(x+1)*size,y*size:(y+1)*size]

    return img_inpt,img_label

def load_imgs_label(data_dir,crop_size,min=None):
    imgs_coord = get_blur_files(data_dir,crop_size)

    if min != None:
        imgs_coord = imgs_coord[:min]
    input = []
    target = []
    i = 0
    for imgtuple in imgs_coord:
        img_inpt,img_label = get_inpt_label(imgtuple,crop_size)
        img_inpt = img_inpt/ (255. / 2.) - 1
        img_label = img_label/(255. / 2.) - 1
        input += [img_inpt]
        target += [img_label]
        if i%100==0:
            print('data load...: '+str(i))
        i+=1
    input_seq = np.asarray(input)
    target_seq = np.asarray(target)
    # cv.namedWindow('a',0)
    # cv.imshow('a',input_seq[0][:])
    # cv.namedWindow('b',0)
    # cv.imshow('b',target_seq[0][:])
    # cv.waitKey(0)
    return input_seq,target_seq
def random_batch(x_data,y_data,batch_size):
    rnd_indices = np.random.randint(0, len(x_data), batch_size)
    x_batch = x_data[rnd_indices][:]
    y_batch = y_data[rnd_indices][:]
    return x_batch, y_batch