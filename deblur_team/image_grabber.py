import os
import cv2
import matplotlib.pyplot as plt


def grab_images(dataset_type='dataset_blurred/architecure', train_test_split=0.9):
    images = list(map(lambda x: dataset_type + "/" + x, os.listdir(dataset_type)))
    train_images = []
    test_images = []
    num_train = int(0.9 * len(images))
    num_test = len(images) - num_train

    for i in range(num_train):
        image = cv2.imread(images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        train_images.append(image)

    for i in range(num_test):
        image = cv2.imread(images[i + num_train])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        test_images.append(image)

    return train_images, test_images

