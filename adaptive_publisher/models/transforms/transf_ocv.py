import numpy as np


import cv2
import numpy as np
import torch


def crop_center(img, size):
    h, w = img.shape[:2]
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    return img[start_h:start_h+size, start_w:start_w+size]

# def crop_center(image, size):
    # h, w = image.shape[:2]
    # crop_size = (size, size)

    # # Calculate starting point for crop
    # start_x = max(0, (w - crop_size[0]) // 2)
    # start_y = max(0, (h - crop_size[1]) // 2)

    # # Perform center crop
    # cropped_image = image[start_y:start_y + crop_size[1], start_x:start_x + crop_size[0]]

    # return cropped_image


def to_tensor(img):
    img_float = img.astype(np.float32)
    # Normalize the image to the range [0, 1]
    img_float /= 255.0
    return img_float


def to_pytorch_tensor_format(img):
    # Transpose to match PyTorch's tensor format (H, W, C) to (C, H, W)
    img_t = np.transpose(img, (2, 0, 1))
    return img_t

def normalize_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    normalized_img = (img - mean) / std
    return normalized_img


def resize(image, target_size):
    # Resize the image to have a fixed shorter edge of target_size while maintaining the aspect ratio
    w, h = image.shape[:2]
    aspect_ratio = h / w

    if aspect_ratio < 1:
        width = target_size
        height = int(width / aspect_ratio)
    else:
        height = target_size
        width = int(height * aspect_ratio)

    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    return resized_image


def obj_transf(image):
    obj_img = resize(image, 640)
    return obj_img


def cls_transf(image):
    cls_img = resize(image, 232)
    cls_img = crop_center(cls_img, 224)
    cls_img = to_tensor(cls_img)
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    cls_img = normalize_image(cls_img, mean, std)
    cls_img = to_pytorch_tensor_format(cls_img)
    return torch.from_numpy(cls_img.astype('float32'))


def get_transforms_ocv(mode='CLS'):
    transform = None
    if mode.upper() == 'OBJ':
        transform = obj_transf
    if mode.upper() == 'CLS':
        transform = cls_transf
    return transform


get_transforms = get_transforms_ocv