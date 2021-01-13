import os
import cv2


DATASET_DIRECTORY = 'dataset/'
UNIFORM_SHAPE = (128,128)


def clean_image_names(path=DATASET_DIRECTORY): 
    '''
        Clean up the image names for easier association between blurred vs un-blurred versions
        @params path <string> Path to the dataset directory, defaults to dataset/
    '''

    dataset_types = list(map(lambda x: DATASET_DIRECTORY+x+'/',os.listdir(DATASET_DIRECTORY)))
    
    # Main loop over all the files in the dataset.
    for type_id, dataset_type in enumerate(dataset_types):
        images = list(map(lambda x: dataset_type+x,os.listdir(dataset_type)))
        # clean up image names for easier comparisons
        for idx, image in enumerate(images):
            new_name = image.split('/')
            new_name[-1] = '{}_{}.jpg'.format(type_id, idx)
            new_name = '/'.join(new_name)        
            os.rename(image, new_name)



def resize_to_default(path=DATASET_DIRECTORY):
    '''
        Resizes all images in the dataset directory to 128x128 pixels
        @params path <string> path to the dataset directory
    '''
    dataset_types = list(map(lambda x: DATASET_DIRECTORY+x+'/',os.listdir(DATASET_DIRECTORY)))
    for type_id, dataset_type in enumerate(dataset_types):
        images = list(map(lambda x: dataset_type+x,os.listdir(dataset_type)))
        # clean up image names for easier comparisons
        for idx, image in enumerate(images):
            im = cv2.imread(image)
            try:
                # resize when shape is not equal
                if im.shape[0] != 128 or im.shape[1] != 128: 
                    print("Resizing {}".format(image))
                    resized = cv2.resize(im, UNIFORM_SHAPE, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(image, resized)
                 
            except:
                os.remove(image)
            

def validate_sizes(path=DATASET_DIRECTORY):
    '''
        Validates that all images in the dataset directory are of same shape (128x128)
        @params path <string> path to the dataset directory
        @returns <Boolean> True if all images are of 128x128 size and false otherwise
    '''

    dataset_types = list(map(lambda x: DATASET_DIRECTORY+x+'/',os.listdir(DATASET_DIRECTORY)))
    for type_id, dataset_type in enumerate(dataset_types):
        images = list(map(lambda x: dataset_type+x,os.listdir(dataset_type)))
        # clean up image names for easier comparisons
        for idx, image in enumerate(images):
            im = cv2.imread(image)
            try:
                if im.shape[0] != 128 or im.shape[1] != 128:
                    return False
            except:
                continue
    
    return True


