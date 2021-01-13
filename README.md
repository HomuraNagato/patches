
## Patches Description

This is an application of the blur-deblur teams (name not final)
neural network (NN) focused deblur algorithms on heterogenous images.
Most deblur algorithms assume an entire image is blurred. This project
aims to deblur portions of an image. This is done in multiple steps.

Preprocess

* Take a pristine image
* Create patches of blur using the fourier transform of the image and a gaussian mask

Task

* Sample random locations in the heterogenous image
* Use a single/suite-of patch(s) to classify if the patch is blurred
* If positive, apply NN to deblur, re-insert into the image
* Repeat for all samples


## Preprocess

The function blur within the class PicOperator takes in a rectangle as a patch,
converts it into fourier space, generates a gaussian mask of the patches shape,
convolves/multiplies the two, followede by transforming back into normal space.
Applied to each RGB channel. Then inserted back into the image.

A pristine image

![A pristine image](images/DSC00462.JPG "pristine image")

A heterogeneous/blurred image

![A heterogenous image](output/DSC00462_blurred.png "heterogenous/blurred image")
DSC00462_blurred.png

## Probability of patch coverage

Given: image size, patch size

Todo: use probability to estimate upper bound on number of patches to cover an image

Once we have a certain number of patches to sample from, we can generate that number
to classify, deblur, and postprocess.

There is quite a bit of probability work in this field. I would say it is a polygon covering
problem. This can be bounded using epsilon-nets and VC-dimension. My probability is weak, so
I hope my understanding is correct. Before that lets apply some simple lower and upper bounds.
Additionally my notation is suspect. R is the rectangle we wish to cover, eg our image, say
4000 x 6000. R' is a patch, say 100 x 100. N are the number of patches we need to to cover R.
A(R) is the area of R. 

### lower bound

Given a rectangle R of size n by m, and a fixed-size patch R' of size x by y, where x < n and
y < m, the minimum number of R' to cover R would be disjoint sets aligned along their borders.
One can compute this using area's:

<img src="https://latex.codecogs.com/svg.latex?\Omega(N)&space;=&space;\frac{A(R)}{A(R')}" title="\Omega(N) = \frac{A(R)}{A(R')}" />

For example, ff R = 16 x 16 and R' is 4 x 4, then we need (16 * 16)/(4 * 4) = 16 patches

So far, not so bad.

### upper bound

Given the same R and R' as before, what are the maximum number of R' to cover R? Assume every R'
is offset from the next by one pixel. We would need a patch for nearly every pixel then.

<img src="https://latex.codecogs.com/svg.latex?O(N)&space;=&space;(m-x)(n-y)" title="O(N) = (m-x)(n-y)" />

For example, R = 16 x 16 and R' is 4 x 4, we would need (16 - 4)**2 = 144 patches

### Experimental setup

In my setup R = 4000 x 6000 and R' = 100 x 100, this gives
<img src="https://latex.codecogs.com/svg.latex?\Omega(N)&space;=&space;2400&space;\text{&space;and&space;}&space;O(N)&space;=&space;2.3*10^6" title="\Omega(N) = 2400 \text{ and } O(N) = 2.3*10^6" />

This is a very large range. Are there ways to reduce our upper bound? Let's investigate.

### Covering number

The upper bound proposed is simply unrealistic. The lower bound is too optimistic.
Is there a way to find a good middle ground? With covering numbers I think there is.

A covering number is the number of sperical balls of a fixed size to completely cover a
given space; overlaps are allowed.

## Classify if an image is blurred

opencv has a mostly straightforward classifier. We can calculate the laplacian of an
image. From my reading, a discrete laplacian is the sum of the second derivatives in a
3 by 3 convolution for each pixel. This is useful to tell us edges and motion of the
image. If an image is in focus, there should be many edges, resulting in a higher variance.
If the image is blurred, there should be fewer edges, and a lower variance. Unfortunately,
this also classifies scenic images (eg of the sky) also as blurry as there are few edges.
A side question I have is, is variance agnostic to the size of the input image? Anyway,
My source used a static value to classify of 100. In my testing, it wasn't performing
how I wanted to, so I normalized in the 'normal' way:
(laplacian_var - laplacian.min()) / (laplacian.max() - laplacian.min())
followed by a 0.25 cutoff. This works on a few of the scenic images I have. So,
good for now.

## Deblur

Pass the blurred patch to a trained algorithm, recieve the deblurred patch.


## Postprocess

Simply apply (best?) deblurred patch back into image. Best is questionable,
best could be first patch that overlaps the most blurred portion and deblurs it, or
two patches that are partially blurred that together deblur the image more. 

Chose to 
