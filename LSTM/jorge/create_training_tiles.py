# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:39:02 2022

@author: jmaie
"""

"""
This file is for preprocessing of the input data. 
Multiple years of population and additional input data sets are stacked.
MODIS land cover data is adjusted to just reflect 4 classes as one-hot layer.
Data is masked to Lima Metropolitan Area.
Image is split into 169 smaller but overlapping tiles for training of the neural networks.
"""

proj_dir = "/Users/jorgemorales/Desktop/Jorge/2024/ModelC/"


import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os


def min_max_scale(img): # (t,c,w,h); channels: lc, pop, urb_dist, slope, streets_dist, water_dist, center_dist, class0mask, class1mask, class2mask, class3mask
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_new = np.zeros(img.shape)
    data_new[:,0,:,:] = img[:,0,:,:] # lc not to be normalized

    for i in range(1, img.shape[1]): # for each feature except lc
        temp = img[:,i,:,:].reshape(-1,1) # combine all axis to one
        scaler.fit(temp)
        new_data = scaler.transform(temp)
        new_data = new_data.reshape(img.shape[0], img.shape[-2], img.shape[-1]) # reshape to raster data
        data_new[:,i,:,:] = new_data

    return np.float32(data_new)

# Visualize a batch of data
def visualize_batch(input, target):
    # Select the first sample in the batch
    input_sample = input  # [5, 10, 256, 256]
    target_sample = target  # [256, 256]
    print(input_sample.shape, target_sample.shape)
    
    # Plot the first feature of the first year
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.title('Input Sample - Year 1, Feature 1')
    plt.imshow(input_sample[0, 0, :, :], cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title('Input Sample - Year 5, Feature 10')
    plt.imshow(input_sample[4, 0, :, :], cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.title('Target Sample')
    plt.imshow(target_sample[0], cmap='viridis')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

# loop over years and stack all the data to retreive an array [20,7,888,888]:
seq = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
n_years = 20
n_classes = 4
dat_multitemp = np.zeros((n_years,7,888,888))

# i=0
# for y in seq:
#    im = io.imread(proj_dir + 'data/yearly_no_na/brick_20' + y + '.tif') # pop, urb_dist_r, lc_r, slope_r, streets_dist_r, water_dist_r, center_dist_r
#    im_move = np.moveaxis(im, 2, 0) # (w,h,c) -> (c, w, h)
#    im_move[[1,2],:,:] = im_move[[2,1],:,:] #switch lc with urban dist
#    im_move[[0,1],:,:] = im_move[[1,0],:,:] #switch lc with pop
#    temp = im_move[3,:,:]
#    temp[temp<0] = 0
#    im_move[3,:,:] = temp
#    dat_multitemp[i,:,:,:] = im_move # stack all yearly stacks together
#    i += 1
dat_multitemp = np.load(proj_dir + "data/ori_data/" + "data.npy")  

print(dat_multitemp.shape) # [:,x,:,:] = (lc_r, pop, urb_dist_r, slope_r, streets_dist_r, water_dist_r, center_dist_r)
print(dat_multitemp[:,0,:,:].max())
plt.imshow(dat_multitemp[1,1,:,:]) 
#
lu_new = dat_multitemp[:,0,:,:]
lu_new[lu_new==1] = 0
lu_new[lu_new==2] = 0
lu_new[lu_new==3] = 0
lu_new[lu_new==4] = 0
lu_new[lu_new==5] = 0
lu_new[lu_new==6] = 0
lu_new[lu_new==8] = 1
lu_new[lu_new==7] = 2
lu_new[lu_new==0] = 3
print(np.unique(lu_new[:, :, :])) # shape: (t, c, w, h)
#
dat_multitemp[:,0,:,:] = lu_new
lcnew_multitemp = dat_multitemp
#
# assign new class values, to have only 4 classes
# lcnew_multitemp = dat_multitemp
# lcnew_multitemp[dat_multitemp == 1] = 0 # ENV -> vegetation
# lcnew_multitemp[dat_multitemp == 2] = 0 # EBV -> vegetation
# lcnew_multitemp[dat_multitemp == 3] = 0 # DNV -> vegetation
# lcnew_multitemp[dat_multitemp == 4] = 0 # DBV -> vegetation
# lcnew_multitemp[dat_multitemp == 5] = 0 # croplands -> vegetation
# lcnew_multitemp[dat_multitemp == 6] = 0 # grassland -> vegetation
# lcnew_multitemp[dat_multitemp == 8] = 1 # urban
# lcnew_multitemp[dat_multitemp == 7] = 2 # barren
# lcnew_multitemp[dat_multitemp == 0] = 3 # water
# np.unique(lcnew_multitemp[:, 0, :, :]) # shape: (t, c, w, h)



# add class masks as input factors
import torch
def oh_code(a, class_n = n_classes):
    oh_list = []
    for i in range(class_n): # for each class
        temp = torch.where(a == i, 1, 0) # binary mask per class
        oh_list.append(temp) # store each class mask as list entry
    return torch.stack(oh_list,0) #torch.stack(oh_list,1) # return array, not list


crop_img_lulc = torch.from_numpy(lcnew_multitemp[:, 0, :, :]) # was not converted to torch before, select lc
temp_list = []
for j in range(crop_img_lulc.shape[0]): # for each year?
    temp = oh_code(crop_img_lulc[j], class_n=n_classes) # array of binary mask per class
    # print(temp.min(), temp.max())
    temp_list.append(temp[np.newaxis, :, :, :]) # store class masks per year in list
oh_crop_img_lulc = np.concatenate(temp_list, axis=0)
oh_crop_img = np.concatenate((crop_img_lulc[:,np.newaxis,:,:], lcnew_multitemp[:, 1:, :, :], oh_crop_img_lulc), axis=1)
# oh_crop_img with lc, pop, urb_dist, slope, streets_dist, water_dist, center_dist, class0mask, class1mask, class2mask, class3mask
oh_crop_img = np.float32(oh_crop_img) # for smaller storage size


#### set everything outsite lima region to 0 -> island will be removed from population
oh_crop_img_lima = oh_crop_img.copy()
# lima = np.load(proj_dir + 'data/ori_data/Lima_region.npy')
for i in range(20):
    # oh_crop_img_lima[i,1,:,:][lima==0] = 0 # pop
    oh_crop_img_lima[i,1,420-1:522-1,78-1:78+96-1] = 0 # pop
    oh_crop_img_lima[i,7,420-1:522-1,78-1:78+96-1] = 0 # bu
    oh_crop_img_lima[i,8,420-1:522-1,78-1:78+96-1] = 0 
    oh_crop_img_lima[i,9,420-1:522-1,78-1:78+96-1] = 0
    oh_crop_img_lima[i,10,420-1:522-1,78-1:78+96-1] = 1
#
    # oh_crop_img_lima[i,3,400:600,:170] = 0 # slope
    # oh_crop_img_lima[i,7,:,:][lima==0] = 0 # class0
    # oh_crop_img_lima[i,8,:,:][lima==0] = 0 # class1
    # oh_crop_img_lima[i,9,:,:][lima==0] = 0 # class2
    # oh_crop_img_lima[i,10,400:600,:170] = 1 # water
        
plt.imshow(oh_crop_img_lima[-1,1,:,:])
#
oh_crop_img_lima[:,1,:,:] = np.where(oh_crop_img_lima[:,1,:,:]<0.5, 
                                         0, oh_crop_img_lima[:,1,:,:])
#
# save unnormed data
np.save(proj_dir + 'data/ori_data/input_all_unnormed.npy', oh_crop_img_lima)


# full_image = np.load(proj_dir + 'data/ori_data/input_all_unnormed.npy')
full_image = oh_crop_img_lima
print(full_image[0][1].min(),full_image[0][1].max(), full_image[0][1].mean())

# normalize values
print('normalizing')
full_image = min_max_scale(full_image) # lc as first channel (t,c,w,h)
print(full_image[0][1].min(),full_image[0][1].max(), full_image[0][1].mean())
np.save(proj_dir + 'data/ori_data/input_all.npy', full_image) # lc unnormed, pop normed, ...

# slice the input data
h_total = full_image.shape[-1] #888
w_total = full_image.shape[-2] #888
img_size = 256 # how big the tiles should be


# // means integer division
x_list = np.linspace(img_size//2, h_total -(img_size//2), num = 10) # Return evenly spaced numbers over a specified interval
y_list = np.linspace(img_size//2, w_total -(img_size//2), num = 10) # num = w_step, num = 14

new_x_list = []
new_y_list = []

for i in x_list: # new list for integers
    for j in y_list:
        new_x_list.append(int(i))
        new_y_list.append(int(j))
# it's like coordinate.


sub_img_list = []

for x, y in zip(new_x_list, new_y_list):
    sub_img = full_image[:, :, x - 128:x + 128, y - 128:y + 128] # get subimage around centroid
    sub_img_list.append(np.float32(sub_img))    
    
print(len(sub_img_list))
print(sub_img_list[1].shape)
#
#
dir_input = proj_dir + 'data/train/input_less_tiles/'
dir_target = proj_dir + 'data/train/target_less_tiles/'
##
ttt = "pop"
#
if ttt == "all":
    dir_input = proj_dir + 'data/train/input_less_tiles_all/'
    dir_target = proj_dir + 'data/train/target_less_tiles_all/'
#
os.makedirs(dir_input, exist_ok=True)
os.makedirs(dir_target, exist_ok=True)

# save all sub images separately
for i in range(len(sub_img_list)):
    #
    visualize_batch(sub_img_list[i][:,1:,:,:], sub_img_list[i][:,1,:,:])
    #
    np.save(dir_input + str(i) + '_input.npy', sub_img_list[i][:,1:,:,:]) # all except lc not normed
    if ttt == "pop":
        np.save(dir_target + str(i) + '_target.npy', sub_img_list[i][:,1,:,:]) # pop normed
    elif ttt == "static":
        np.save(dir_target + str(i) + '_target.npy', sub_img_list[i][:,[1,3,4,5,6],:,:]) # pop normed
    else:
        np.save(dir_target + str(i) + '_target.npy', sub_img_list[i][:,[1,2,7,8,9,10,3,4,5,6],:,:]) # pop normed
    ## from---> lc, pop, urb_dist, slope, streets_dist, water_dist, center_dist, class0mask, class1mask, class2mask, class3mask
    ## to---> pop, urb_dist, class0mask, class1mask, class2mask, class3mask, slope, streets_dist, water_dist, center_dist
    #
    print(f"tile {i}")
    print(sub_img_list[i].shape)
    print(sub_img_list[i][:,1,:,:].min(), sub_img_list[i][:,1,:,:].max())   
#
tempi = np.array(sub_img_list)
print(tempi.min(), tempi.max(), tempi.mean())



# for i in range(16):
#     inp = np.load(proj_dir + 'data/train/input_less_tiles/'+ str(i) +'_input.npy')
#     plt.imshow(inp[0,0,:,:])
#     plt.show()

# targ = np.load(proj_dir + 'data/train/target_less_tiles/10_target.npy')



