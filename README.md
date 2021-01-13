
## Patches Description

This is an application of the blur-deblur teams
neural network (NN) focused deblur algorithms on heterogenous images.
Most deblur algorithms assume an entire image is blurred. This project
aims to deblur portions of an image. This is done in multiple steps.

Preprocess

* Take a pristine image
* Create patches of blur using the Fourier transform of the image and a gaussian mask

Task

* Sample random locations in the heterogenous image
* Create a patch from a sampled location
* Classify if the patch is blurred
* If positive, apply NN to deblur then re-insert into the image
* Repeat for all samples


## Apply blurs

The function blur within the class PicOperator takes in a rectangle as a patch,
converts it into Fourier space, generates a gaussian mask of the patches shape,
convolves/multiplies the two, followed by transforming back into normal space.
Applied to each RGB channel. Then inserted back into the image. This is done in
select locations in the image.

A pristine image

![A pristine image](images/DSC00462.JPG "pristine image")

A heterogeneous/blurred image

![A heterogenous image](output/img_de-blurred_blocks_result_DSC00462.png "heterogenous/blurred image")


## Classify if an image is blurred

Opencv has a mostly straightforward classifier. We can calculate the laplacian of an
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

## Probability of patch coverage

Given: image size, patch size

This was the most interesting part of the project for me.

To start off lets apply some simple lower and upper bounds to our project.
Let us say that R is the rectangle we wish to cover, say 4000 x 6000.
R' is a patch, say 100 x 100. N are the number of patches we need to to cover R.
A(R) is the area of R.

### lower bound

Given a rectangle R of size n by m, and a fixed-size patch R' of size x by y, where x < n and
y < m, the minimum number of R' to cover R would be disjoint sets aligned along their borders.
One can compute this using area's:

<img src="https://latex.codecogs.com/svg.latex?\Omega(N)&space;=&space;\frac{A(R)}{A(R')}" title="\Omega(N) = \frac{A(R)}{A(R')}" />

For example, if R = 4000 x 6000 and R' is 100 x 100, then we need 2,400 patches.

This would be too optimistic I thought. As the probability that if we sampled 2,400
times, they would all align in the way just described, seemed quite unlikely.

### upper bound

Given the same R and R' as before, what are the maximum number of R' to cover R? Assume every R'
is offset from the next by one pixel. We would need a patch for nearly every pixel then.

<img src="https://latex.codecogs.com/svg.latex?O(N)&space;=&space;(m-x)(n-y)" title="O(N) = (m-x)(n-y)" />

For example, R = 4000 x 6000 and R' is 100 x 100, we would need 23 million patches. This is
quite large and I thought I could do better.

### Covering number

The upper bound proposed is simply unrealistic. The lower bound is too optimistic.
Is there a way to find a good middle ground? With covering numbers I think there is.

A covering number is the number of sperical balls of a fixed size to completely cover a
given space; overlaps are allowed. We want to solve the equation below.
Definitions:
 - <img src="https://latex.codecogs.com/svg.latex?N^{ext}_{r}(K)" title="N^{ext}_{r}(K)" /> = External covering number
 - K = Euclidian space containing a set of vectors/balls
 - k = length (norm) of any vector in K is at most k
 - d = dimensional space, 2
 - r = radius of fixed size ball

<img src="https://latex.codecogs.com/svg.latex?N^{ext}_{r}(K)&space;\leq&space;(\frac{2k&space;\sqrt{d}}{r})^d" title="N^{ext}_{r}(K) \leq (\frac{2k \sqrt{d}}{r})^d" />

Now with a bit of imagination, if we say the diagonal of our image is k (the largest vector within K), d is most
definitely 2, and r is the diagonal of our cover image divided by two (to get a radius), then we have everything we
need to calculate a covering number! In my example this becomes 83,200.


## Postprocess

Simply apply (best?) deblurred patch back into image. Best is questionable,
best could be first patch that overlaps the most blurred portion and deblurs it, or
two patches that are partially blurred that together deblur the image more. I chose
for a real time update in the image, as I wasn't convinced a more sophisticated
method would do any better.

## Deblur

Now that we know how many patches to sample, have a way to classify it, and run it through
a deblurring algorithm, we can finally run the program! Below is the result. One can see
artifacts from the algorithm. At least to me, it appears the regions that are explicitly
blurred were targeted more, though we already know it's not the best. Also It's apparent
the deblur algorithm could be improved. To have made it this far in only two weeks seems commendable.

![final image](output/img_de-blurred_covering_DSC00462.png "final 'deblurred' image")
