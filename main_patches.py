
import numpy as np
import argparse
import cv2

from skimage import restoration
from scipy.signal.signaltools import wiener

import sys
from main import main, parse_args

import json
import h5py
# start: 7:41pm
# end: 8:30?
# lower took 4 min 20 sec 260 seconds
# covering: 50781
# lower: 1464
# (50781 / 1464 * 260) / 60 = 150
# 20310000 / 1464 * 260) / 60 = 42 days

def initialization():
    # argparse
    # - in and out paths?
    # - rectangles, best way if need variable number and shape?
    parser = argparse.ArgumentParser(
        description="deblur an image by a patches",
        epilog="example: python main_patches.py --image images/DSC00462.JPG"
    )
    parser.add_argument('--image', type=str, required=False, default='images/DSC00462.JPG',
                        help='image to act on')
    parser.add_argument('--outdir', type=str, required=False, default='output',
                        help='directory to store any saved images')

    args = parser.parse_args()
    print(args)
    return args

def matlab_gauss(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    source here: 
    https://pythonpedia.com/en/knowledge-base/17190649/how-to-obtain-a-gaussian-filter-in-python
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

class Rectangle(object):
    def __init__(self, *coord):
        # coord = y0, y1, x0, x1
        assert(len(coord) == 4), "a rectangle requires four corners"
        
        self.y0 = coord[0]
        self.y1 = coord[1]
        self.x0 = coord[2]
        self.x1 = coord[3]
        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0
        self.area = self.width * self.height
        
        assert (self.y0 >= 0), "overlay vertical should be positive"
        assert (self.y1 >= 0), "overlay horizontal should be positive"
        assert (self.y1 > self.y0), "coord y is invalid"# by association, self.y1 > 0
        assert (self.x1 > self.x0), "coord x is invalid"

    def bound_check(self, shape):
        assert (self.y1 <= shape[0]), "overlay vertical out of bounds"
        assert (self.x1 <= shape[1]), "overlay horizontal is out of bounds"

    def __str__(self):
        return "({}, {}, {}, {})".format(self.y0, self.y1, self.x0, self.x1)

class PicOperator(object):

    def __init__(self, img_path):
        self.I = cv2.imread(img_path)
        self.height = self.I.shape[0]
        self.width = self.I.shape[1]
        self.area = self.height * self.width

    def post_process(self):
        # postprocess for saving
        self.I = np.abs(self.I)
        #I -= I.min() # necessary? possibly img has no black?
        self.I = self.I * 255
        self.I = self.I.astype(np.uint8)

    def blur(self, rect=None):
        """
        blur self.I based on coordinates in rect

        :rect: Rectangle object with coordinates to blur within image
        """
        # matlab code
        # https://stackoverflow.com/questions/26744673/low-pass-filtering-a-color-image-using-the-fft-and-ifft

        # blur the entire image if not defined
        if not rect:
            rect = Rectangle(0, self.I.shape[0], 0, self.I.shape[1])

        rect.bound_check(self.I.shape)
        red   = self.I[rect.y0:rect.y1, rect.x0:rect.x1, 0]
        green = self.I[rect.y0:rect.y1, rect.x0:rect.x1, 1]
        blue  = self.I[rect.y0:rect.y1, rect.x0:rect.x1, 2]
        shape = (rect.y1-rect.y0, rect.x1-rect.x0)

        # transform and shift
        f_r = np.fft.fft2(red)
        fs_r = np.fft.fftshift(f_r)

        f_g = np.fft.fft2(green)
        fs_g = np.fft.fftshift(f_g)

        f_b = np.fft.fft2(blue)
        fs_b = np.fft.fftshift(f_b)

        # prep gaussian filter
        gauss = matlab_gauss(shape=shape, sigma=20)
        gauss = np.fft.fft2(gauss)
        gauss = np.fft.fftshift(gauss)

        # filter
        ff_r = fs_r * gauss
        ff_g = fs_g * gauss
        ff_b = fs_b * gauss

        # inverse back and cast to real
        Ir = np.fft.ifftshift(ff_r)
        Irr = np.fft.fftshift(np.real(np.fft.ifft2(Ir)))

        Ig = np.fft.ifftshift(ff_g)
        Igg = np.fft.fftshift(np.real(np.fft.ifft2(Ig)))

        Ib = np.fft.ifftshift(ff_b)
        Ibb = np.fft.fftshift(np.real(np.fft.ifft2(Ib)))

        # last step, insert blur
        iblur = cv2.merge([Irr, Igg, Ibb])
        # any constraints or checks? separate function?
        self.I[rect.y0:rect.y1, rect.x0:rect.x1, :] = iblur

    def trim(self, r):
        """convert rect pointers into a rectangle in the image"""
        return self.I[r.y0:r.y1, r.x0:r.x1, :]
        
    def save(self, fout, r=None, I=None):
        """save the image, or a rectangle of the image"""
        if not r:
            cv2.imwrite(fout, self.I)
        else:
            cv2.imwrite(fout, self.trim(r))
            
    def insert(self, r, img):
        assert (r.height == img.shape[0]), "mismatch between insert and position in I"
        assert (r.width == img.shape[1]), "mismatch between insert and position in I"
        self.I[r.y0:r.y1, r.x0:r.x1, :] = img

    def sample_patches(self, n, r):
        """
        :n:  number of patches to generate
        :r: rectangle patch to determine area of available pixels to draw from
        :res: list of patches drawn
        """
        inner_height = self.height - r.height
        inner_width = self.width - r.width
        samples_h = np.random.choice(inner_height, n)
        samples_w = np.random.choice(inner_width, n)
        res = []
        for i in range(n):
            res.append(Rectangle(samples_h[i], samples_h[i] + r.height,
                                 samples_w[i], samples_w[i] + r.width))
        return res

    def lower_bound(self, r):
        """
        calculate the minimum number of patches to cover I
        view readme for insight - A(R)/A(R')
        """
        return int(self.area/r.area)

    def upper_bound(self, r):
        """
        calculate maximum number of patches to cover I
        view readme for insight - (y1-y0)*(x1-x0) ~ n^2
        """
        return (self.height - r.height) * (self.width - r.width)

    def covering_bound(self, r):
        """
        K: set of vectors in R**m
        k: K's norm is at most k. essentially the dimensions of our image
        d: d-dimensional subspace of R**m, in euclidean == 2
        r: radius of fixed size balls
        :res: external_covering_number
        """
        k = np.sqrt(self.width**2 + self.height**2)
        d = 2
        r = np.sqrt(r.width**2 + r.height**2) / 2
        return int((2*k*np.sqrt(d)/r)**d)

    def rademacher_bound(self, r):
        # c = k?   m = r?
        # this is find, this is actually, relatively, a large number
        # examples have it less than 1?
        c = np.sqrt(self.width**2 + self.height**2)
        d = 2
        m = np.sqrt(r.width**2 + r.height**2) / 2
        return c * np.sqrt(d * np.log(d)) / m

        
    def variance_of_laplacian(self, I):
        """
        variance of laplacian represents the focus of the image
        if above our threshold, then classify image as blurry
        required to be in greyscale?
        """
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(I, cv2.CV_64F).var(), cv2.Laplacian(I, cv2.CV_64F)
    
    def is_blurred(self, I, epsilon = 100, verbose=False):
        """
        :epsilon: hyperparameter for classifying image as blurred or not
        :res: boolean whether image is blurred or not
        source:
        https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
        """
        lacianv, laplacian = self.variance_of_laplacian(I)
        normalized_lacianv = lacianv/laplacian.max()
        if verbose:
            print("image laplacian variance: {} normalized: {}".format(lacianv, normalized_lacianv))
        if lacianv < epsilon and normalized_lacianv < 0.25:
            return True
        else:
            return False

    def deblur_by_patches(self, patches):
        """
        :patches: rects within I to
           - test if blurred
           - apply deblur
           - insert into I
        """
        n_blurred = 0
        for patch in patches:
            I_patch = self.trim(patch)
            blurred = self.is_blurred(I_patch)

            if blurred:
                n_blurred += 1
                deblurred_patch = main(ARGS, I=I_patch)
                deblurred_patch = (deblurred_patch * 255).astype(np.uint8)
                self.insert(patch, deblurred_patch)
        
        print("number of patches defined blurred: {}/{}".format(n_blurred, len(patches)))

    

if __name__ == '__main__':

    args = initialization()
    ARGS = parse_args()
    
    # blur image
    fpath = args.image
    fname = fpath.split('/')[-1][:-4]
    print("fpath {}  fname: {}".format(fpath, fname))
    Iobj = PicOperator(fpath)

    # y0, y1, x0, x1
    rect1 = Rectangle(2100, 2500, 2100,2500) # 400 x 400             false  X
    rect2 = Rectangle(1000, 1500, 1000, 1600) # 500 x 600            true   O 
    rect3 = Rectangle(1300, 1900, 3000, 4000) # 600 x 1000           true   O
    rect4 = Rectangle(2100, 3100, 3100, 4500) # 1000 x 1400          false  X
    rect5 = Rectangle(2000, 2100, 2000, 2100) # pristine region      false  O
    rect6 = Rectangle(1450, 1550, 1550, 1650) # corner of rect2      true   O?
    rect7 = Rectangle(1000, 1128, 1000, 1128) # inner rect1          false  X
    rect8 = Rectangle(1128, 1256, 1128, 1256) # more inner rect1     true   O
    rect9 = Rectangle(1256, 1384, 1256, 1384) # more more inner rect1 false X
    # 5/9 correctly classified
    Iblur = Iobj.blur(rect1)
    Iblur = Iobj.blur(rect2)
    Iblur = Iobj.blur(rect3)
    Iblur = Iobj.blur(rect4)
    Iobj.save("{}/img_blurred_{}.png".format(args.outdir, fname))

    # sample patches from image
    cover = Rectangle(0, 128, 0, 128)
    n_lower = Iobj.lower_bound(cover)
    n_upper = Iobj.upper_bound(cover)
    n_covering = Iobj.covering_bound(cover)
    n_rademacher = Iobj.rademacher_bound(cover)
    print("the optimal number of sample patches is between ({}, {}) ".format(n_lower, n_upper))
    print("external covering number: {} rademacher bound: {}".format(n_covering, n_rademacher))

    #patches = Iobj.sample_patches(n_lower, cover)
    #Iobj.deblur_by_patches(patches)

    # test on classifying blurred

    #test = [rect1, rect2, rect3, rect4, rect5, rect6, rect7, rect8, rect9]
    """
    test = [rect7, rect8, rect9]
    for i, r in enumerate(test):
        patch = Iobj.trim(r)
        print("{} is blurred? {}".format(r, Iobj.is_blurred(patch, verbose=False)))

        cover_img = main(ARGS, I=patch)
        #cover_img = np.zeros(patch.shape)
        cover_img = (cover_img * 255).astype(np.uint8)
        print("i diff: {}".format(np.sum(patch-cover_img)))
        
        Iobj.insert(r, cover_img)

    """
    #Iobj.save("{}/img_de-blurred_lower_{}.png".format(args.outdir, fname))
