# Similar to Nehme et al.

import numpy as np
from os.path import abspath
from skimage import io

# Help function from other impementation
def project_01(im):
    im = np.squeeze(im)
    min_val = im.min()
    max_val = im.max()
    return (im - min_val)/(max_val - min_val)

# normalize image given mean and std
def normalize_im(im, dmean, dstd):
    im = np.squeeze(im)
    im_norm = np.zeros(im.shape,dtype=np.float32)
    im_norm = (im - dmean)/dstd
    return im_norm

def getTest(version="ours"):
    np.random.seed(123)

    ##### Generate training dataset
    datafile_list = [abspath("benchmark_data/testStack_SimulatedMicrotubules.tif")]

    image_list = []
    for datafile in datafile_list:

        mean_std = np.load("models/" + version + "_meanstd.npy")
        mean_val_test, std_val_test = mean_std[0], mean_std[1]

        # load the tiff data
        Images = io.imread(datafile)

        # Only difference with Nehme et al. is that we do not perform any upsampling of the input image, since
        # our loss supports working on small resolution images.
        Images = Images

        # upsampled frames dimensions
        (K, M, N) = Images.shape

        # Setting type
        Images = Images.astype('float32')

        # Normalize each sample by it's own mean and std
        Images_norm = np.zeros(Images.shape, dtype=np.float32)
        for i in range(Images.shape[0]):
            Images_norm[i, :, :] = project_01(Images[i, :, :])
            Images_norm[i, :, :] = normalize_im(Images_norm[i, :, :], mean_val_test, std_val_test)

        # Reshaping
        Images_norm = np.expand_dims(Images_norm, axis=3)

        image_list.append(Images_norm)


    return image_list
