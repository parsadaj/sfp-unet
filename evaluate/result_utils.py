import os
import numpy as np


def crop_to_bbox(object: dict, bbox_margin=0):
    mask = object['mask']
    bbox = get_bounding_box(mask)
    
    new_object = {}
    
    for k, v in object.items():
        if k in ['Normals_gt', 'images', 'mask']:
            new_object[k] = apply_bounding_box(v, bbox, bbox_margin)
    
    return new_object



def predict_one_object(model, object, deghat=0, patch_size = 64):
    if deghat == 0:
        stride = int(patch_size / 2)
        counter_step = 10
    else:
        stride = int(patch_size / 4)
        counter_step = 10
            
    normals = object['Normals_gt']
    images = object['images']
    mask = object['mask']
    
    normals = np.pad(normals, ((patch_size, patch_size), (patch_size, patch_size), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    images = np.pad(images, ((patch_size, patch_size), (patch_size, patch_size), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    mask = np.pad(mask, ((patch_size, patch_size), (patch_size, patch_size)), 'constant', constant_values=((0, 0), (0, 0)))
    
    new_predicted_image = np.zeros_like(normals)
    count_predicted_image = np.zeros(normals.shape[:-1])
    
    i_range = range(0, 1024+128-patch_size, stride)
    j_range = range(0, 1224+128-patch_size, stride)

    counter = 0
    total = len(j_range) * len(i_range)

    for i in i_range:
        for j in j_range:
            counter += 1
            if counter % counter_step == 0:
                print("{} / {}".format(counter, total), end='\r')
            current_patch = images[np.newaxis, i:i+patch_size, j:j+patch_size]
            current_mask = mask[i:i+patch_size, j:j+patch_size]
            if np.sum(current_mask) == 0:
                continue
            predict_current = model.predict(current_patch, verbose=0)[0, ..., :]
            new_predicted_image[i:i+patch_size, j:j+patch_size, :] += predict_current
            count_predicted_image[i:i+patch_size, j:j+patch_size] += 1

    count_predicted_image[count_predicted_image == 0] = 1
    normals_hat = new_predicted_image.copy() / count_predicted_image[..., np.newaxis]
    
    norm = np.linalg.norm(normals_hat, axis=2, keepdims=True)
    norm[norm==0] = 1

    normals_hat = normals_hat / norm
    normals_hat[mask==0] = 0
    return normals_hat[64:64+1024, 64:64+1224]


def calculate_mae(pred: np.ndarray, label: np.ndarray):
    adotb = np.sum(pred * label, axis=1, keepdims=True)
    norma = np.linalg.norm(pred, axis=1, keepdims=True)
    normb = np.linalg.norm(label, axis=1, keepdims=True)
    
    norma[norma == 0] = 1
    normb[normb == 0] = 1
    
    ae = np.arccos(adotb / norma / normb)
    return np.mean(ae)


def get_bounding_box(mask):
    return 0, mask.shape[0], 0, mask.shape[1]
    

def apply_bounding_box(image, bbox, margin=0):
    i_start, i_end, j_start, j_end = bbox
    return image[i_start-margin:i_end+margin, j_start-margin:j_end+margin, ...]


def predict_one_object2(model, object, deghat=0):
    if deghat == 0:
        stride = 16
        counter_step = 10
    else:
        stride = 8
        counter_step = 10
        
    patch_size = 32
    
    normals = object['Normals_gt']
    images = object['images']
    mask = object['mask']
    
    normals = np.pad(normals, ((patch_size, patch_size), (patch_size, patch_size), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    images = np.pad(images, ((patch_size, patch_size), (patch_size, patch_size), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    mask = np.pad(mask, ((patch_size, patch_size), (patch_size, patch_size)), 'constant', constant_values=((0, 0), (0, 0)))
    
    new_predicted_image = np.zeros_like(normals)
    count_predicted_image = np.zeros(normals.shape[:-1])
    
    i_range = range(0, 1024+128-patch_size, stride)
    j_range = range(0, 1224+128-patch_size, stride)

    counter = 0
    total = len(j_range) * len(i_range)

    for i in i_range:
        for j in j_range:
            counter += 1
            if counter % counter_step == 0:
                print("{} / {}".format(counter, total), end='\r')
            current_patch = images[np.newaxis, i:i+patch_size, j:j+patch_size]
            current_mask = mask[i:i+patch_size, j:j+patch_size]
            if np.sum(current_mask) == 0:
                continue
            predict_current = model.predict(current_patch, verbose=0)[0, ..., :]
            new_predicted_image[i:i+patch_size, j:j+patch_size, :] += predict_current
            count_predicted_image[i:i+patch_size, j:j+patch_size] += 1

    normals_hat = new_predicted_image.copy() / count_predicted_image[..., np.newaxis]

    normals_hat = normals_hat / np.linalg.norm(normals_hat, axis=2, keepdims=True)
    normals_hat[mask==0] = 0
    return normals_hat[32:32+1024, 32:32+1224]


def create_if_not_exist(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        

def predict_one_object_onelook(model, obj):
      
    patch_size = 64
    
    normals = obj['Normals_gt']
    images = obj['images']
    mask = obj['mask']
    
    pad_h = 128
    pad_w = 58
    
    normals = np.pad(normals, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    images = np.pad(images, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=((0, 0), (0, 0)))
    

    normals_hat = model.predict(images[np.newaxis, ...], verbose=0)[0, ..., :]
    normals_hat = normals_hat / np.linalg.norm(normals_hat, axis=2, keepdims=True)
    normals_hat[mask==0] = 0
    return normals_hat[pad_h:pad_h+1024, pad_w:pad_w+1224]
