import cv2
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def GrayGradient(image):  ### 用的这个
	# image.size=[128,96,65,65]
	t1=time.time()
	# count_batch,count_channel,count_heigt,count_width = image.size()
	b, c, h, w = image.size()
	# # print("count_width  count_heigt count_channel",count_width,count_heigt,count_channel)
	idx_batch = []
	for k in range(b):
		idx_batch0 = [k]*c
		idx_batch = idx_batch+idx_batch0
	idx_channel0 = [c for c in range(0,c)]
	idx_channel = idx_channel0*b
	t2=time.time()
	# print(len(idx_channel),"\n",t2-t1)
	#[12288,64,65]

	image1 = image
	image_x = F.pad(image1, pad=(1,0,0,0),mode='constant')
	image_y = F.pad(image1, pad=(0,0,1,0),mode='constant')
	gradient_x = image_x[idx_batch,idx_channel, :,1:h+1] - image_x[idx_batch,idx_channel,:, 0:h]
	gradient_y = image_y[idx_batch,idx_channel, 1:w+1, :] - image_y[idx_batch,idx_channel, 0:w, :]
	img_AG_pix = torch.pow((torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))/2, 0.5)




	# dw = torch.pow((image[idx_batch,idx_channel,1:,:]-image[idx_batch,idx_channel,:count_width-1,:]),2)
	# # [12288,65,64]
	# dh = torch.pow((image[idx_batch,idx_channel,:,1:]-image[idx_batch,idx_channel,:,:count_heigt-1]),2)
	#
	# dh = torch.transpose(dh,dim0=1,dim1=2)
	# #[12288,64,65]

	img_AG_h = torch.sum(img_AG_pix,dim=1,keepdim=True)
	img_AG_w= torch.sum(img_AG_h,dim=2,keepdim=True)
	list_bc = [bc for bc in range(c*b)]
	#img_AGradient = [12288,1.1]
	img_AGradient0= img_AG_w[list_bc,:,:]/float((h)*(w))+0.00001
	img_AGradient = img_AGradient0.view(b,c,1,1)

	AGray = nn.AdaptiveAvgPool2d((1,1))
	img_AGray = AGray(image.float())

	img_GG = img_AGray/img_AGradient
	# img_GG = img_GG.view(b, c, 1, 1)
	t3 = time.time()
	# print("all_time",t3-t1)
	return img_GG,img_AGray,img_AGradient

#梯度
def GrayGradientori(image):
	# image.size=[128,96,65,65]
	# print(image.size())
	# width_img = image.size()[3]
	count_width = image.size()[3]
	# heigt_img = image.size()[2]
	count_heigt = image.size()[2]
	count_channel  = image.size()[1]
	# print("count_width  count_heigt count_channel",count_width,count_heigt,count_channel)
	# print("width_img",width_img)
	img = image.view([-1,1,count_heigt,count_width])
	# print(img.shape)
	img_chw = np.squeeze(img,1)
	# print(img_chw.shape)
	count_b_c =  img_chw.size()[0]
	imageAG_chw = torch.ones(count_b_c,1,1)
	# print("imageAG_chw",imageAG_chw.shape)
	for cc in range(count_b_c):
		img_hw = img_chw[cc]
		dw = torch.pow((img_hw[1:,:]-img_hw[count_width-1:,:]),2).sum()
		dh = torch.pow((img_hw[:,1:]-img_hw[:,count_heigt-1:]),2).sum()
		img_hw_AG = torch.pow((dw+dh)/2,0.5)
		float_img_hw = img_hw.float()
		img_hw_Agray = torch.sum(float_img_hw)
		# print(img_hw,"\n")
		# print(img_hw_Agray,"\n")
		# print("img_hw_AG",img_hw_AG)
		if img_hw_Agray<0.01:
			imageAG_chw[cc, :, :] =0
		else:
			imageAG_chw[cc,:,:]=img_hw_Agray/img_hw_AG
		# print(imageAG_chw)
		l=1
	# print(imageAG_chw)
	imageAG = torch.unsqueeze(imageAG_chw,1)
	# print(imageAG.shape)
	imageAG_ = imageAG.view([-1,count_channel,1,1])
	# print("imageAG_",imageAG_.shape,imageAG_.type)
	# print(imageAG_)
	return imageAG_

#空间频率SF
def spatialF(image):
	M = image.shape[0]
	N = image.shape[1]
	
	cf = 0
	rf = 0


	for i in range(1,M-1):
		for j in range(1,N-1):
			dx = float(image[i,j-1])-float(image[i,j])
			rf += dx**2
			dy = float(image[i-1,j])-float(image[i,j])
			cf += dy**2

	RF = math.sqrt(rf/(M*N))
	CF = math.sqrt(cf/(M*N))
	SF = math.sqrt(RF**2+CF**2)

	return SF

# #
if __name__ == "__main__":
	img_ori = "/home/yj02/pycharm-community-2020.2.5/pycharm_projects/lyh_projects/RED-CNN-master/dataset/NUDT-SIRST/images/000025.png"
	img_noise = "/home/yj02/pycharm-community-2020.2.5/pycharm_projects/lyh_projects/RED-CNN-master/dataset/NUDT-SIRST-Noise_v1/images/000025.png"
	img_denoise = "/home/yj02/pycharm-community-2020.2.5/pycharm_projects/lyh_projects/RED-CNN-master/results/NUDT-SIRST-Noise_v1/REDNet30_65_2022_04_28_22_39_08-NUDT-SIRST_v1/test/images/000025.png"
	img_ori,img_noise,img_denoise = cv2.imread(img_ori), cv2.imread(img_noise),cv2.imread(img_denoise)

	img_ori = torch.from_numpy(img_ori)
	img_noise = torch.from_numpy(img_noise)
	img_denoise = torch.from_numpy(img_denoise)

	img_ori = img_ori.permute([2, 0, 1])[:1, :, :]
	img_noise = img_noise.permute([2, 0, 1])[:1, :, :]
	img_denoise = img_denoise.permute([2, 0, 1])[:1, :, :]

	img_ori = torch.unsqueeze(img_ori,0)
	img_noise = torch.unsqueeze(img_noise, 0)
	img_denoise = torch.unsqueeze(img_denoise, 0)
	print(img_denoise.shape)
	GrayGradient(img_denoise)