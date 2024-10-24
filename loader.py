import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from PIL import Image, ImageOps, ImageFilter

class ct_dataset(Dataset):

    def __init__(self, mode, data_path, input_path,target_path,mask_path,patch_n, patch_size, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"

        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform
        self.data_path = data_path
        self.input_path  = input_path
        self.target_path = target_path
        self.mask_path = mask_path

        #all images
        # x=noise_img=input_img,y=clean_img= target_img
        input_path = os.path.join(data_path,input_path)
        target_path = os.path.join(data_path, target_path)
        mask_path = os.path.join(data_path, mask_path)

        #select from mode.txt
        txt_path = os.path.join(data_path,mode)+'.txt'
        fid = open(txt_path, 'r+')
        fid_content = fid.readlines()
        fid.close()
        input_path = [os.path.join(input_path, i.split()[0] + '.png') for i in fid_content]
        target_path = [os.path.join(target_path, i.split()[0] + '.png') for i in fid_content]
        mask_path = [os.path.join(mask_path, i.split()[0] + '.png') for i in fid_content]
        input_ = [f for f in input_path]
        target_ = [f for f in target_path]
        mask_ = [f for f in mask_path]
        self.input_ = input_
        self.target_ = target_
        self.mask_ = mask_

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        # x=noise_img=input_img,y=clean_img= target_img
        input_img, target_img ,mask_img= self.input_[idx], self.target_[idx],self.mask_[idx]
        # print(input_img)
       ###input_img = 'D:/paper_project/IRDnNet/dataset\\NUDT-SIRST-Noise_v3/images\\001142.png'
      #  im_name = input_img.split('/')[-1]
        im_name = input_img.split('\\')[-1]
        # print(input_img, target_img)

        input_img, target_img,mask_img = cv2.imread(input_img), cv2.imread(target_img),cv2.imread(mask_img)

        # img  = Image.open(input_img).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        # mask = Image.open(target_img)

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        if self.patch_size:
            input_patches, target_patches, mask_patches = get_patch(input_img,
                                                      target_img,
                                                      mask_img,
                                                      self.patch_n,
                                                      self.patch_size)

            # print(len(input_patches),input_patches[0].shape, len( mask_patches),mask_patches[0].shape)

            return (input_patches, target_patches, mask_patches)
        else:
            input_img = torch.from_numpy(input_img)
            target_img = torch.from_numpy(target_img)
            mask_img = torch.from_numpy(mask_img)
            input_img = input_img.permute([2,0,1])[:1,:,:]
            target_img = target_img.permute([2, 0, 1])[:1, :, :]
            mask_img = mask_img.permute([2, 0, 1])[:1, :, :]
            return (input_img, target_img, mask_img,im_name)


def get_patch_ori(full_input_img, full_target_img, patch_n, patch_size):

    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w , c = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        if h-new_h >0:
            top = np.random.randint(0, h - new_h)
        else:
            top = 0

        if w-new_w>0:
            left = np.random.randint(0, w-new_w)
        else:
            left=0
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w, :]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w, :]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
        # print(1)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)

def get_patch(full_input_img, full_target_img, full_mask_img,patch_n, patch_size):

    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    mask_patches = []
    h, w , c = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    overlap_factor = 2

    if h-new_h >0:
        num_length = int(h /(new_h -2))
    else:
        num_length =1
    if w-new_w>0:
        num_width = int(h / (new_w - 2))
    else:
        num_width = 1

    for i in range(0,num_width):
        for j in range(0,num_length):

            patch_input_img = full_input_img[i*(new_w)-i*overlap_factor:(i+1)*(new_w)-i*overlap_factor,
                              j*(new_w)-j*overlap_factor:(j+1)*(new_w)-j*overlap_factor]
            patch_input_img = cv2.cvtColor(patch_input_img,cv2.COLOR_BGR2GRAY)
            patch_mask_img = full_mask_img[i*(new_w)-i*overlap_factor:(i+1)*(new_w)-i*overlap_factor,
                              j*(new_w)-j*overlap_factor:(j+1)*(new_w)-j*overlap_factor]
            patch_mask_img = cv2.cvtColor(patch_mask_img, cv2.COLOR_BGR2GRAY)
            patch_target_img = full_target_img[i*(new_w)-i*overlap_factor:(i+1)*(new_w)-i*overlap_factor,
                              j*(new_w)-j*overlap_factor:(j+1)*(new_w)-j*overlap_factor]
            patch_target_img = cv2.cvtColor(patch_target_img, cv2.COLOR_BGR2GRAY)
            patch_input_imgs.append(patch_input_img)
            patch_target_imgs.append(patch_target_img)
            mask_patches.append(patch_mask_img)
            # print(patch_mask_img.shape)

            # patch_input_imgs=patch_input_img
            # patch_target_imgs=patch_target_img
            # mask_patches=patch_mask_img
            # if j!=0:
            #     patch_input_imgs=torch.cat(patch_input_imgs,patch_input_img)
            #     patch_target_imgs=torch.cat(patch_target_imgs,patch_target_img)
            #     mask_patches=torch.cat(mask_patches,patch_mask_img)

    # print(mask_patches,'\n',len(mask_patches),"\n",mask_patches[0].shape)

    return np.array(patch_input_imgs), np.array(patch_target_imgs), np.array(mask_patches)

def get_loader(mode,data_path,
               input_path, target_path,mask_path,
               patch_n, patch_size,
               transform=None, batch_size=64, num_workers=1):

    if mode=='test'or mode=='testori'or mode=='testimg':

        dataset_val = ct_dataset('test', data_path, input_path, target_path, mask_path,patch_n, False, transform)
        dataset_train = ct_dataset('train', data_path, input_path, target_path,mask_path, patch_n, False, transform)
        val_data_loader = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False, num_workers=num_workers)
        train_data_loader = DataLoader(dataset=dataset_train, batch_size=1, shuffle=False,
                                       num_workers=num_workers)

    else:
        dataset_val = ct_dataset('test', data_path, input_path, target_path, mask_path,patch_n, False, transform)
        dataset_train = ct_dataset('train', data_path, input_path, target_path, mask_path,patch_n, patch_size, transform)
        val_data_loader = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False, num_workers=num_workers)
        train_data_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return val_data_loader,train_data_loader
