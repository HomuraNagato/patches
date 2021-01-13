import cv2
import os
import random
import numpy as np

DATASET_DIRECTORY = 'dataset/'

def mean_blur(img, kernel_size=0):
    if kernel_size <= 0:
        kernel_size = random.randint(3,8)
        kernel = (kernel_size, kernel_size)
    blured = cv2.blur(img, kernel)
    
    return blured
    

def gaussian_blur(img, kernel_size=0, std=-1):
    if std == -1:
        std = random.randint(2,15)
    
    if kernel_size <= 0:
        kernel_size = random.randrange(3, 9+1, 2)
        kernel = (kernel_size, kernel_size)
    
    blurred = cv2.GaussianBlur(img, kernel, std)
    return blurred

    
def median_blur(img, kernel_size=0):
    if kernel_size <= 0:
        kernel_size = random.randrange(3, 9+1, 2)
    
    blurred = cv2.medianBlur(img, kernel_size)
    return blurred


def bilateral_blur(img, kernel_size=0, std_a=0, std_b=0):
    if kernel_size <= 0:
        kernel_size = random.randint(3,8)
    
    if std_a <= 0: 
        std_a = random.randint(1, 150)
    
    if std_b <= 0: 
        std_b = random.randint(1, 150)
        
    blurred = cv2.bilateralFilter(img, kernel_size, std_a, std_b)
    return blurred


def motion_blur(image, degree=-1, angle=-1):
    if degree <= -1:
        degree = random.randint(1, 45)
    
    if angle <= -1:
        angle = random.randint(1, 45)
    
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    
    return blurred

def normalize(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    img = np.array(img, dtype=np.uint8)
    return img

# Test blurring
def test():
    image = '../dataset/architecure/3_23.jpg'
    
    img = cv2.imread(image)
    cv2.imshow('Source image', img)
    
    mean_blurred = mean_blur(img)
    gaussian_blurred = gaussian_blur(img)
    median_blurred = median_blur(img)
    bilateral_blurred = bilateral_blur(img)
    motion_blurred = motion_blur(img)
    
    cv2.imshow('Mean Blur', normalize(mean_blurred))
    cv2.imshow('Gaussian Blur', normalize(gaussian_blurred))
    cv2.imshow('Median Blur', normalize(median_blurred))
    cv2.imshow('Bilateral Blur', normalize(bilateral_blurred))
    cv2.imshow('Motion Blur', normalize(motion_blurred))
    
    cv2.waitKey()


def create_blurred_dataset(dataset=DATASET_DIRECTORY):
    # Folder exists
    if not os.path.exists('dataset_blurred/'):
        os.mkdir('dataset_blurred/')    
    else:
        print('Directory Exists!')
    
    dataset_types = list(map(lambda x: DATASET_DIRECTORY+x+'/',os.listdir(DATASET_DIRECTORY)))


    for type_id, dataset_type in enumerate(dataset_types):
        dataset_subset = 'dataset_blurred/' + dataset_type.split('/')[-2]

        
        
        if not os.path.exists(dataset_subset):
            os.mkdir(dataset_subset)

        images = list(map(lambda x: dataset_type+x,os.listdir(dataset_type)))
        
        for idx, image in enumerate(images):
            image_name = image.split('/')[-1]
            blurred_image_path = dataset_subset + '/' + image_name
            blur = random.randint(1, 5)
            
            img = cv2.imread(image)
            
            cv2.waitKey()    
            if blur == 1:
                blurred = mean_blur(img)
                cv2.imwrite(blurred_image_path, blurred)
            elif blur == 2: 
                blurred = gaussian_blur(img)
                cv2.imwrite(blurred_image_path, blurred)
            elif blur == 3:
                blurred = median_blur(img)
                cv2.imwrite(blurred_image_path, blurred)
            elif blur == 4: 
                blurred = bilateral_blur(img)
                cv2.imwrite(blurred_image_path, blurred)
            elif blur == 5:
                blurred = motion_blur(img)
                cv2.imwrite(blurred_image_path, blurred)
            else:
                print('Invalid Blur type {} specified'.format(blur))
            
            