import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.optimize

import PIL
from PIL import Image
from PIL import ImageEnhance
import skimage.transform

# Fit Gaussian to region
def fit_region(img, x_location, y_location):

    def ff(pp, x0,x1,x2):
        smoothing = 5
        var = multivariate_normal(mean=[x0, x1], cov=[[smoothing, 0], [0, smoothing]])
        return x2 * var.pdf(pp) * ((2 * np.pi) * smoothing)

    scope=6 #6
    xmax = img[max(x_location-scope,0):x_location+scope, max(y_location-scope,0):y_location+scope].shape[0]
    ymax = img[max(x_location-scope,0):x_location+scope, max(y_location-scope,0):y_location+scope].shape[1]
    xx, yy = np.mgrid[0:xmax, 0:ymax]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    parameter, _ = scipy.optimize.curve_fit(ff,positions.T,
                                            img[max(x_location-scope,0):x_location+scope, max(y_location-scope,0):y_location+scope].flatten(),
                                            p0=[xmax/2,ymax/2,1],
                                            maxfev=200)

    return parameter, [max(y_location-scope,0)+parameter[1],max(x_location-scope,0)+parameter[0]]

# Locations to heatmap
def generate_heatmap(img, location):
    xmax = img.shape[0]
    ymax = img.shape[1]
    xx, yy = np.mgrid[0:xmax, 0:ymax]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    smoothing = 5
    var = multivariate_normal(mean=np.flipud(location), cov=[[smoothing, 0], [0, smoothing]])
    return np.reshape(var.pdf(positions.T), xx.shape) * ((2 * np.pi) * smoothing)

# Subpixel refinement
def subpixelPrediction(preds,count_max=100):

    plt.figure()
    plt.imshow(preds)
    plt.colorbar()
    plt.savefig("tmp.png")
    plt.close()

    prediction = np.copy(preds)

    location_list = []
    count=0
    while True and count<count_max:
        value = np.max(prediction.flatten())
        if value<0.4:
            break
        x, y = np.where(prediction==value)

        for kk in range(x.shape[0]):
            try:
                parameter, location = fit_region(prediction, x[kk], y[kk])
            except:
                print('optimization failed')
                parameter =[0,0,1]
                location = [y[kk], x[kk]]
            if parameter[2]>0.2 and parameter[2]<1.5:
                location_list.append(location)
                count += 1

            prediction-=generate_heatmap(prediction, location)

    if count==count_max:
        print("maximum reached...", count, count_max)
        plt.figure()
        plt.imshow(preds)
        plt.colorbar()
        plt.savefig("tmp.png")
        plt.close()
    else:
        print("Max. not reached: ", count, count_max)


    return location_list

# Argmax refinement
def argmaxPrediction(prediction,count_max=100):

    prediction = np.copy(prediction)
    location_list = []
    count=0
    while True or count<count_max:
        value = np.max(prediction.flatten())
        if value<0.4: #thresholding
            break
        x, y = np.where(prediction==value)

        for kk in range(x.shape[0]):
            location_list.append([y[kk], x[kk]])
            prediction-=generate_heatmap(prediction, [y[kk], x[kk]])
            count+=1

    if count==count_max:
        print("maximum reached...")


    return location_list


# Apply transformation to image
def random_transform(img):
    img = Image.fromarray(img)

    factor = 5
    converter = PIL.ImageEnhance.Color(img)  # color balance
    img2 = converter.enhance(max(min(1 + np.random.randn() / factor, 2), 0.25))

    converter = PIL.ImageEnhance.Contrast(img2)  # contrast
    img3 = converter.enhance(max(min(1 + np.random.randn() / factor, 2), 0.25))

    converter = PIL.ImageEnhance.Brightness(img3)  # brightness
    img4 = converter.enhance(max(min(1 + np.random.randn() / factor, 2), 0.25))

    converter = PIL.ImageEnhance.Sharpness(img4)  # Sharpness
    img5 = converter.enhance(max(min(1 + np.random.randn() / factor, 2), 0.25))

    # Greyscale
    if np.random.rand() > 0.9:
        img6 = np.tile(np.array(img5.convert('LA'))[:,:,0:1],[1,1,3])
    else:
        img6 = np.array(img5)


    # Flip
    if np.random.rand() > 0.5:
        img6 = np.transpose(img6,[1,0,2])
    if np.random.rand() > 0.5:
        img6 = np.fliplr(img6)

    if np.random.rand() > 0.5:
        return np.array( np.round(skimage.transform.rotate(img6.astype(np.float), angle=30 * (np.random.rand() - 0.5), mode='reflect')).astype(np.uint8))
    else:
        return np.array(img6)

