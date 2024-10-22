import argparse
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from .encoder_runner import run_encoder
from .model import DeblurringCNN
from .create_blur_dataset import create_blurred_dataset

import json
import h5py

def fix_layer0(filename, batch_input_shape, dtype):
    print(filename)
    with h5py.File(filename, 'r+') as f:
        raw = f.keys()
        print("fix_layer0 raw\n{}".format(raw))
        model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
        layer0 = model_config['config'][0]['config']
        layer0['batch_input_shape'] = batch_input_shape
        layer0['dtype'] = dtype
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')

def parse_args():
    
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's deblur some images!!"
    )
    parser.add_argument(
        '--model',
        action = 'store',
        help = '''Pass in the model you want to use: sf, cnn, enc'''
    )
    parser.add_argument(
        '--encoder',
        action = 'store',
        help = '''Pass in the type of encoder you want to use: auto, blur, var'''
    )
    parser.add_argument(
        '--generate-dataset',
        action = 'store_true',
        help = '''Pass this argument if you haven't generated the dataset yet '''
    )
    parser.add_argument(
        '--predict_all',
        action = 'store_true',
        help = '''Pass this argument if you have a saved model and want to use it to deblur all images in the dataset '''
    )
    parser.add_argument(
        '--predict',
        action = 'store',
        help = '''Pass in the path to the single image you want to deblur'''
    )
    return parser.parse_args()

def get_images():
    #For now, taking only architecture photos
    gauss_blur = os.listdir('dataset_blurred')
    sharp = os.listdir('dataset')
    x_blur = []
    y_sharp = []
    for i in tqdm(range(len(gauss_blur))):
        gauss_blur_2 = os.listdir('dataset_blurred/' + gauss_blur[i])
        sharp_2 = os.listdir('dataset/' + sharp[i])
        for j in range(len(gauss_blur_2)):
            x_blur.append(cv2.imread('dataset_blurred/'+str(gauss_blur[i]) +'/' + str(gauss_blur_2[j]))/255.0)
            y_sharp.append(cv2.imread('dataset/'+str(sharp[i]) +'/' + str(sharp_2[j]))/255.0)
    return np.asarray(x_blur, dtype = np.float32), np.asarray(y_sharp, dtype=np.float32)

def main(ARGS, I=[]):
    
    if ARGS.generate_dataset:
        create_blurred_dataset()

    model_type = ARGS.model
    predict_path = ARGS.predict

    if ARGS.predict_all:
        model = DeblurringCNN()
        model.load_weights('model.h5')
        model.compile()

        if not os.path.exists('results/cnn/'):
            os.mkdir('results/cnn/')

        dataset_types = list(map(lambda x: 'dataset_blurred/'+x+'/',os.listdir('dataset_blurred/')))
        for type_id, dataset_type in enumerate(dataset_types):
            dataset_subset = 'results/cnn/' + dataset_type.split('/')[-2]
            if not os.path.exists(dataset_subset):
                os.mkdir(dataset_subset)

            images = list(map(lambda x: dataset_type+x,os.listdir(dataset_type)))
            
            for idx, image in enumerate(images):
                image_name = image.split('/')[-1]
                deblurred_image_path = dataset_subset + '/' + image_name
                fig = plt.figure()
                img = cv2.imread(image)
                fig.add_subplot(1,2,1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img = img/255.0
                img = np.asarray([img])
                deblurred = model.predict(img)
                deblurred = deblurred[0]
                fig.add_subplot(1,2,2)
                plt.imshow(cv2.cvtColor(deblurred, cv2.COLOR_BGR2RGB))
                plt.savefig(deblurred_image_path)
                plt.close()
        exit(0)


    if (predict_path != None) or (len(I) > 0):
        model = DeblurringCNN()
        #fix_layer0('model.h5', [128,128,3], 'float32')
        #print("init model: {}".format(model))
        x_train = np.asarray(cv2.imread(predict_path), dtype = np.float32)
        y_train = np.asarray(cv2.imread(predict_path), dtype = np.float32)
        #model.fit(x_train, y_train, epochs=0)
        #print("model fit (supposedly)")
        model.build((None,128,128,3))
        model.load_weights('model.h5')
        model.compile()

        # override with path
        if predict_path != None:
            blurred = cv2.imread(str(predict_path))
        else:
            blurred = I
        
        if blurred is None:
            print("invalid filepath")
            exit(0)
        if not(len(blurred.shape) == 3 and blurred.shape[0] == 128 and blurred.shape[1]==128 and blurred.shape[2]==3):
            print("Wrong dims")
            exit(0)
        #cv2.imshow("blurred", blurred)
        #cv2.waitKey()
        blurred = blurred/ 255.0
        blurred = np.asarray([blurred])
        deblurred = model.predict(blurred)
        deblurred = deblurred[0]
        #cv2.imshow("deblurred", deblurred)
        #cv2.waitKey()
        #exit(0)
        return deblurred

    if model_type == 'sf':
        #Write something so it doesn't do all images?
        if not os.path.exists('results/sharpen/'):
            os.mkdir('results/sharpen/')

        dataset_types = list(map(lambda x: 'dataset_blurred/'+x+'/',os.listdir('dataset_blurred/')))
        for type_id, dataset_type in enumerate(dataset_types):
            dataset_subset = 'results/sharpen/' + dataset_type.split('/')[-2]
            if not os.path.exists(dataset_subset):
                os.mkdir(dataset_subset)

            images = list(map(lambda x: dataset_type+x,os.listdir(dataset_type)))
            
            for idx, image in enumerate(images):
                image_name = image.split('/')[-1]
                sharpened_image_path = dataset_subset + '/' + image_name
                img = cv2.imread(image)
                sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #Can be played around with
                sharpened = cv2.filter2D(img, -1, sharpen_kernel)
                cv2.imwrite(sharpened_image_path, sharpened)

    elif model_type == 'cnn':
        model = DeblurringCNN()
        x_train, y_train = get_images()
        # (x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.25)
        model.compile()
        # print(model.model.summary())
        model.fit(x_train, y_train, epochs= 40, batch_size=3)
        
        # model.compile(model.optimizer, loss = 'mean_squared_error', metrics = ['mean_squared_error'])
        # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.001)
        # model.fit(x_train, y_train, batch_size = batch, epochs = 40, validation_data = (x_val, y_val), callbacks = [reduce_lr])
        model.save_weights('model.h5')

    elif model_type == 'enc':
        encoder_type = ARGS.encoder
        run_encoder(encoder_type)

    else:
        print("Invalid model argument specified. Valid arguments are: sf, cnn, enc.")
        exit(1)

if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)

