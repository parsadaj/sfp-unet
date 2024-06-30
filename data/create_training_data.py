import os
import scipy.io
import numpy as np
from ctd_utils import create_folder_if_not_exist, extract_patches
import rasterio

parent_folder = os.getcwd() # Path to Parent Folder
data_folder = os.path.join(parent_folder, 'data')
dataset_folder_name = 'dataset'

test_lists_path = os.path.join(data_folder, dataset_folder_name, 'test_list.csv')
with open(test_lists_path) as f:
    test_list = f.read().split('\n')[:-1]
    
train_lists_path = os.path.join(data_folder, dataset_folder_name, 'train_list.csv')
with open(train_lists_path) as f:
    train_list = f.read().split('\n')[:-1]

        

for mode in ['train', 'test']:
    create_folder_if_not_exist(os.path.join(data_folder, mode))
    create_folder_if_not_exist(os.path.join(data_folder, mode, 'x'))
    create_folder_if_not_exist(os.path.join(data_folder, mode, 'y'))
    create_folder_if_not_exist(os.path.join(data_folder, mode, 'w'))

test_out_path = os.path.join(data_folder, 'test')
train_out_path = os.path.join(data_folder, 'train')

patch_size = 64
stride = 32

nodata_percent_to_remove = 0.15
max_allowed_nodata = patch_size * patch_size * nodata_percent_to_remove


image_group_names = 'x', 'y', 'w'
mode = 'train', 'test'
out_paths = train_out_path, test_out_path

counter = 0
total_count = 0
count_poo = 0

for iiii, relpaths_list in enumerate([train_list, test_list]):
    tile_num = 0

    for relpath in relpaths_list:
        fullpath = os.path.join(data_folder, dataset_folder_name, *relpath.split('/'))
        mat = scipy.io.loadmat(fullpath)
        images = mat['images']
        normals = mat['Normals_gt']
        mask = mat['mask']
        
        image_tiles = extract_patches(images, patch_size=patch_size, stride=stride)
        normals_patches = extract_patches(normals, patch_size=patch_size, stride=stride)
        noDataRef_tiles = extract_patches(mask, patch_size=patch_size, stride=stride)
        noDataValue = 0
        

        total_num_tiles = image_tiles.shape[0]
        total_count += total_num_tiles
        count_poo += 1

        for i in range(total_num_tiles):
            ## skip tiles with large no-data parts
            count_nodata = np.count_nonzero(noDataRef_tiles[i, ...] == noDataValue)
            counter += 1
            
            if counter % 1000 == 0:
                print("{}: {} / {}".format(count_poo, counter, total_count), end='\r')
                
            if count_nodata < max_allowed_nodata:
                ## save results
                # tile_name = str(tile_num) + '_' + pasvand +'.tif'
                tile_name = str(tile_num) + relpath.replace('/','_') + '.tif'
                
                
                out_image_path = os.path.join(out_paths[iiii], image_group_names[0], tile_name)

                x_out = np.rollaxis(image_tiles[i, ...], 2, 0).copy()
                w_out = np.rollaxis(noDataRef_tiles[i, ...], 2, 0).copy()
                y_out = np.rollaxis(normals_patches[i, ...], 2, 0).copy()
                
                
                x_out[:, w_out[0]==0] = 0
                y_out[:, w_out[0]==0] = 0
                w_out = np.vstack([w_out, w_out, w_out])
                
                with rasterio.open(
                    out_image_path,
                    'w',
                    driver='GTiff',
                    height=x_out.shape[1],
                    width=x_out.shape[2],
                    count=x_out.shape[0],
                    dtype=rasterio.float32,
                    compress="lzw"
                ) as dst:
                    dst.write(x_out)
                    
                out_image_path = os.path.join(out_paths[iiii], image_group_names[1], tile_name)
                
                with rasterio.open(
                    out_image_path,
                    'w',
                    driver='GTiff',
                    height=y_out.shape[1],
                    width=y_out.shape[2],
                    count=y_out.shape[0],
                    dtype=rasterio.float32,
                    compress="lzw"
                ) as dst:
                    dst.write(y_out)

                out_image_path = os.path.join(out_paths[iiii], image_group_names[2], tile_name)
                with rasterio.open(
                    out_image_path,
                    'w',
                    driver='GTiff',
                    height=w_out.shape[1],
                    width=w_out.shape[2],
                    count=w_out.shape[0],
                    dtype=rasterio.float32,
                    compress="lzw"
                ) as dst:
                    dst.write(w_out)

                tile_num += 1

    
    

    
    