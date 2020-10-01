####### Generate checkerboard

import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
from scipy.stats import multivariate_normal
import sklearn.datasets
import glob
import os

import PIL
from PIL import Image
from PIL import ImageEnhance

from scipy.spatial import ConvexHull
from multiprocessing import Pool

_bool_texture=True
_generateTrainData = True # Train or Test samples

## Create directories to save images
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("testData"):
    os.makedirs("testData")

def generateRandomCheckerboard(idx):
    np.random.seed(idx)

    ### Crop image
    def crop_image(img, points):
        x_min = max(np.min(np.where(np.sum(img, axis=(1, 2)) != 0)) - 1, 0)
        x_max = np.max(np.where(np.sum(img, axis=(1, 2)) != 0)) + 1
        y_min = max(np.min(np.where(np.sum(img, axis=(0, 2)) != 0)) - 1, 0)
        y_max = np.max(np.where(np.sum(img, axis=(0, 2)) != 0)) + 1

        cropped = img[x_min:x_max, y_min:y_max]
        points -= np.array([y_min, x_min])[np.newaxis, np.newaxis, :]

        return cropped, points

    ### Rotation
    def randomRotate(checkerboard, real_corners):
        height, width = checkerboard.shape[:2]
        image_center = (width / 2, height / 2)

        M = cv2.getRotationMatrix2D(image_center, np.random.randint(360), 1.)

        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])

        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        M[0, 2] += bound_w / 2 - image_center[0]
        M[1, 2] += bound_h / 2 - image_center[1]

        warped = cv2.warpAffine(checkerboard, M, (bound_w, bound_h))

        warped_points = \
            np.dot(
                np.concatenate([real_corners[:, 0, :], np.ones([real_corners.shape[0], 1]).astype(np.float32)], axis=1),
                np.transpose(M)).astype(np.float32)
        warped_points = warped_points[:, np.newaxis, :]

        return crop_image(warped, warped_points)

    ### Perspective Transform
    def randomPerspectiveTransform(checkerboard, real_corners, dim):
        original = real_corners[
            [0, dim[1] - 2, (dim[0] - 1) * (dim[1] - 1) - dim[1] + 1, (dim[0] - 1) * (dim[1] - 1) - 1]]
        flag = 0
        while flag == 0:
            # print(flag)
            projected = original + np.max(
                scipy.spatial.distance.cdist(original[:, 0, :], original[:, 0, :])) / 15 * np.random.randn(
                original.shape[0], original.shape[1], original.shape[2]).astype(np.float32)
            if np.min(projected) > margin + square_size / 5 \
                    and np.max(projected[:, 0, 0]) < checkerboard.shape[1] - (margin) \
                    and np.max(projected[:, 0, 1]) < checkerboard.shape[0] - (margin) \
                    and ConvexHull(projected[:, 0, :]).simplices.shape[0] == 4:

                hull = ConvexHull(original[:, 0, :])
                hull2 = ConvexHull(projected[:, 0, :])
                mtrx1 = np.concatenate(
                    [np.sort(original[hull.simplices, 0, 0], axis=1), np.sort(original[hull.simplices, 0, 1], axis=1)],
                    axis=1)
                mtrx2 = np.concatenate(
                    [np.sort(original[hull2.simplices, 0, 0], axis=1),
                     np.sort(original[hull2.simplices, 0, 1], axis=1)],
                    axis=1)
                distance = np.zeros([hull.simplices.shape[0], hull2.simplices.shape[0]])
                for kk in range(hull.simplices.shape[0]):
                    for jj in range(hull2.simplices.shape[0]):
                        distance[kk, jj] = np.linalg.norm(mtrx1[kk, :] - mtrx2[jj, :])

                # Same ordering of corner (no crossing)
                if np.sum(np.prod(distance, 1)) == 0 and np.sum(np.prod(distance, 0)) == 0:
                    flag = 1

        if False:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(checkerboard)
            for kk in range(original.shape[0]):
                plt.scatter(original[kk, 0, 0], original[kk, 0, 1])

            for simplex in hull.simplices:
                plt.plot(original[simplex, 0, 0], original[simplex, 0, 1], 'k-')

            plt.subplot(1, 2, 2)
            plt.imshow(checkerboard)
            for kk in range(projected.shape[0]):
                plt.scatter(projected[kk, 0, 0], projected[kk, 0, 1])
            for simplex in hull2.simplices:
                plt.plot(projected[simplex, 0, 0], projected[simplex, 0, 1], 'k-')

        M = cv2.getPerspectiveTransform(original, projected)
        warped = cv2.warpPerspective(checkerboard, M, (checkerboard.shape[1], checkerboard.shape[0]))
        warped_points = cv2.perspectiveTransform(real_corners, M)

        return crop_image(warped, warped_points)

    ### Distotion (maps points using inverse transformation maps)
    def distortion(warped, warped_points, dim, tree=True):

        def mapPoints(real_corners, mapx, mapy, tree=True):
            from sklearn.neighbors import KDTree
            undist_coords_f = np.concatenate([mapy.flatten()[:, np.newaxis], mapx.flatten()[:, np.newaxis]], axis=1)
            if tree:  # More efficient if many points to map
                print('Initialize KDtree')
                tree = KDTree(undist_coords_f)
                print('--Done')

            def calc_val(point_pos, shape_y):
                if tree:
                    nearest_dist, nearest_ind = tree.query([point_pos], k=5)
                    if nearest_dist[0][0] == 0:
                        return undist_coords_f[nearest_ind[0][0]]
                else:
                    dist = np.linalg.norm(undist_coords_f - np.array(point_pos)[np.newaxis, :], axis=1)
                    nearest_ind = np.argpartition(-dist, -5)[-5:]
                    nearest_dist = dist[nearest_ind]

                    idx_sort = np.argsort(nearest_dist)
                    nearest_dist = nearest_dist[idx_sort]
                    nearest_ind = nearest_ind[idx_sort]

                # starts inverse distance weighting
                w = np.array([1.0 / pow(d + 5e-10, 2) for d in nearest_dist])
                sw = np.sum(w)
                x_arr = np.floor(nearest_ind[0] / shape_y)
                y_arr = (nearest_ind[0] % shape_y)
                xx = np.sum(w * x_arr) / sw
                yy = np.sum(w * y_arr) / sw
                return (xx, yy)

            new_corners = np.zeros(real_corners.shape, np.float32)
            for kk in range(real_corners.shape[0]):
                new_corners[kk, 0, [1, 0]] = calc_val([real_corners[kk, 0, 1], real_corners[kk, 0, 0]], mapy.shape[1])

            return new_corners

        # Distortion
        objp = np.zeros(((dim[1] - 1) * (dim[0] - 1), 3), np.float32)
        objp[:, :2] = np.mgrid[0:(dim[1] - 1), 0:(dim[0] - 1)].T.reshape(-1, 2)

        _, mtx, _, _, _ = cv2.calibrateCamera([objp], [warped_points],
                                              cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).shape[::-1], None, None)

        factor = 10
        mtx[0, 2] = factor * np.random.randint(margin, warped.shape[1] - margin)
        mtx[1, 2] = factor * np.random.randint(margin, warped.shape[0] - margin)
        mtx[0, 0] = factor * 1000
        mtx[1, 1] = factor * 1000 * warped.shape[0] / warped.shape[1]

        dist = np.array([30, 0, 0, 0, 0])

        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx,
                                                 (factor * warped.shape[1], factor * warped.shape[0]), 5)

        scaled_board, scaled_point = scalingImage(warped, warped_points, 10)

        dst = cv2.remap(scaled_board, mapx, mapy, cv2.INTER_LINEAR)
        dst_point = mapPoints(scaled_point, mapx, mapy, tree=False)

        scaled_board, scaled_point = scalingImage(addBlur(dst, 1), dst_point, 1 / 10)

        return crop_image(scaled_board, scaled_point)

    ### Lighting
    def addlighting(checkerboard):
        light = 0
        for kk in range(5):
            X, Y = np.meshgrid(np.arange(checkerboard.shape[1]), np.arange(checkerboard.shape[0]))
            pos = np.dstack((X, Y))
            mu = np.array([np.random.randint(checkerboard.shape[1]), np.random.randint(checkerboard.shape[0])])
            cov = checkerboard.shape[0] * checkerboard.shape[1] * sklearn.datasets.make_spd_matrix(2,
                                                                                                   random_state=None) / np.random.randint(
                4, 10)
            rv = multivariate_normal(mu, cov)
            Z = rv.pdf(pos)

            light += (Z - np.min(Z)) * np.random.randint(50, 100) / (np.max(Z) - np.min(Z))

        light = np.minimum(light, 200)

        if False:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.imshow(light)
            plt.colorbar()
            fig.show()

        checkerboard_light = checkerboard.astype(np.float) - (200 - light)[:, :, np.newaxis]
        checkerboard_light = np.round(np.maximum(checkerboard_light, 0)).astype(np.uint8)

        return checkerboard_light

    ### Blur
    def addBlur(checkerboard, blur_level=None):
        if blur_level is None:
            blur_level = np.random.uniform(0, 2)
        # print(blur_level)
        warped_tmp = checkerboard + 5 * blur_level * np.random.randn(checkerboard.shape[0], checkerboard.shape[1],
                                                                     checkerboard.shape[2])  # noise
        warped = np.minimum(np.maximum(warped_tmp, 0), 255).astype(np.uint8)
        warped = cv2.GaussianBlur(warped, (7, 7), np.random.uniform(1, 2) * blur_level)  # blur

        return warped

    ### Sharpness
    def enhanceShapness(warped):
        im = Image.fromarray(warped)
        converter = PIL.ImageEnhance.Sharpness(im)  # Sharpness
        return np.array(converter.enhance(np.random.uniform(0, 2)))

    ### Contrast
    def enhanceContrast(warped):
        im = Image.fromarray(warped)
        converter = PIL.ImageEnhance.Contrast(im)  # Contrast
        return np.array(converter.enhance(np.random.uniform(0.2, 1.2)))

    ### Image scaling
    def scalingImage(checkerboard, real_corners, scaling=None):
        if scaling is None:
            scaling = np.random.uniform(0.5, 1.2, size=2)
        if not hasattr(scaling, "__len__"):
            scaling = (scaling, scaling)

        original = np.array([[100, 100], [200, 100], [100, 200], [200, 200]], np.float32)[:, np.newaxis, :]
        projected = np.copy(original)
        projected[:, :, 0] *= scaling[0]
        projected[:, :, 1] *= scaling[1]

        M = cv2.getPerspectiveTransform(original.astype(np.float32), projected.astype(np.float32))
        warped = cv2.warpPerspective(checkerboard, M, (
            int(np.ceil(scaling[0] * checkerboard.shape[1])), int(np.ceil(scaling[1] * checkerboard.shape[0]))))
        warped_points = cv2.perspectiveTransform(real_corners, M)

        return warped, warped_points


    # Random shape and size
    dim = [np.random.randint(3,7),np.random.randint(3,7)]
    square_size = np.random.randint(15,60) #10,40
    margin = np.random.randint(20,50)

    # Generate Chessboard
    white = np.random.randint(150,255)
    black = np.random.randint(0,white-150) #100
    original_40x40 = np.tile(np.kron([[white, black] * 20, [black, white] * 20] * 20, np.ones((square_size, square_size)))[:,:,np.newaxis],[1,1,3]).astype(np.uint8)

    xx, yy = np.meshgrid(square_size*np.arange(1,dim[1])+margin-0.5, square_size*np.arange(1,dim[0])+margin-0.5)
    real_corners = np.concatenate([xx.flatten()[:,np.newaxis,np.newaxis], yy.flatten()[:,np.newaxis,np.newaxis]],axis=2).astype(np.float32)


    checkerboard = white*np.ones([square_size*dim[0]+2*margin, square_size*dim[1]+2*margin,3], np.uint8)
    checkerboard[margin:square_size*dim[0]+margin, margin:square_size*dim[1]+margin,:] = original_40x40[:square_size*dim[0],:square_size*dim[1],:]

    checkerboard[:,:,0]= np.minimum(np.maximum(checkerboard[:,:,0].astype(np.float) + np.random.randint(-5,5),0),255).astype(np.uint8)
    checkerboard[:,:,1]= np.minimum(np.maximum(checkerboard[:,:,1].astype(np.float) + np.random.randint(-5,5),0),255).astype(np.uint8)
    checkerboard[:,:,2]= np.minimum(np.maximum(checkerboard[:,:,2].astype(np.float) + np.random.randint(-5,5),0),255).astype(np.uint8)

    def perspective(img, points): return randomPerspectiveTransform(img, points, dim)
    def rotate(img, points): return randomRotate(img, points)
    def light(img, points): return (addlighting(img),points)
    def distorte(img, points): return distortion(img, points, dim, tree=False)
    def blur(img, points): return (addBlur(img), points)
    def sharp(img, points): return (enhanceShapness(img), points)
    def contrast(img, points): return (enhanceContrast(img), points)
    def scale(img, points): return scalingImage(img, points)

    transformation = [blur, blur, light, sharp, contrast, scale, distorte, perspective, rotate]
    try:
        if _bool_texture:

            # Load texture image
            files_list = glob.glob('Checkerboard/background/texture*')
            texture_image = np.random.choice(files_list)

            img = cv2.imread(texture_image)
            texture_upsampling = np.ceil(
                max(checkerboard.shape[0] / img.shape[0], checkerboard.shape[1] / img.shape[1]))
            img = cv2.resize(img, (-1, -1), fx=texture_upsampling, fy=texture_upsampling)

            xmin = np.random.randint(img.shape[0]-checkerboard.shape[0])
            ymin = np.random.randint(img.shape[1]-checkerboard.shape[1])
            img = img[xmin:xmin+checkerboard.shape[0], ymin:ymin+checkerboard.shape[1]]

            img = img.astype(np.float32) - np.min(img)
            intensity = np.random.randint(10,50)
            img = img / np.max(img) * intensity - np.random.randint(0,intensity)


            checkerboard= np.maximum(np.minimum(np.round(checkerboard.astype(np.float)+img),255),0).astype(np.uint8)



        #Apply transformation
        warped = np.copy(checkerboard)
        warped_points = np.copy(real_corners)
        for transform_idx in np.random.choice(np.arange(0,len(transformation)), size=np.random.randint(1,len(transformation)), replace=False):
            warped, warped_points = transformation[transform_idx](warped, warped_points)


        # Labels to heatmaps
        xmax = warped.shape[0]
        ymax = warped.shape[1]
        xx, yy = np.mgrid[0:xmax, 0:ymax]
        positions = np.vstack([xx.ravel(), yy.ravel()])

        f = 0
        smoothing = 5
        for kk in range(warped_points.shape[0]):
            var = multivariate_normal(mean=np.flipud(warped_points[kk,0,:]), cov=[[smoothing,0],[0,smoothing]])
            f+= np.reshape(var.pdf(positions.T), xx.shape) * ((2*np.pi)*smoothing)

        if False:
            plt.figure()
            plt.imshow(warped)
            plt.imshow(f, alpha=0.5,cmap='Blues')
            plt.axis('equal')


        data = np.zeros([400,400,3], np.uint8)
        labels = np.zeros([400,400], np.float32)

        data[:warped.shape[0],:warped.shape[1]] = warped
        labels[:f.shape[0],:f.shape[1]] = f

        if False:
            plt.figure()
            plt.imshow(data)
            plt.imshow(labels, alpha=0.5,cmap='Blues')
            plt.axis('equal')

        # Save imagse
        if _generateTrainData:
            cv2.imwrite('data/{}.png'.format(idx), data.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
            np.save('data/{}_data.npy'.format(idx), data.astype(np.uint8))
            np.save('data/{}_label.npy'.format(idx), labels)
            np.save('data/{}_corners.npy'.format(idx), warped_points)
        else:
            cv2.imwrite('testData/{}.png'.format(idx), data.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
            np.save('testData/{}_data.npy'.format(idx), data.astype(np.uint8))
            np.save('testData/{}_label.npy'.format(idx), labels)
            np.save('testData/{}_corners.npy'.format(idx), warped_points)
            np.save('testData/{}_dim.npy'.format(idx), np.array(dim))

    except:
        print(idx, ' failed')



# Function that either generate or load the test dataset
def generateDataset(n_samples, fromScratch=True):

    if fromScratch:
        p = Pool(8)
        p.map(generateRandomCheckerboard, [x for x in np.random.randint(1000000,size=n_samples)])
        p.close()

        generateDataset(n_samples, fromScratch=False)

    else:
        import glob
        files_list = glob.glob('data/*.png')
        np.random.shuffle(files_list)

        data = np.zeros([n_samples, 400,400,3], np.uint8)
        y_data = np.zeros([n_samples, 400,400], np.float32)
        y_loc = []
        for kk in range(len(files_list[:n_samples])):
            data[kk] = np.load(files_list[kk].replace('.png','_data.npy')).astype(np.uint8)
            y_data[kk] = np.load(files_list[kk].replace('.png','_label.npy')).astype(np.float32)
            y_loc.append(np.load(files_list[kk].replace('.png','_corners.npy')).astype(np.float32))

        return data[:len(files_list[:n_samples])], y_data[:len(files_list[:n_samples])], y_loc


# Function that loads the test dataset
def testDataset():

    import glob
    files_list = glob.glob('testData/*.png')
    n_samples = len(files_list)

    data = np.zeros([n_samples, 400,400,3], np.uint8)
    y_loc = []
    dimension = []
    for kk in range(len(files_list[:n_samples])):
        data[kk] = np.load(files_list[kk].replace('.png','_data.npy')).astype(np.uint8)
        y_loc.append(np.load(files_list[kk].replace('.png','_corners.npy')).astype(np.float32))
        dimension.append(np.load(files_list[kk].replace('.png','_dim.npy')).astype(np.float32))

    return data[:len(files_list[:n_samples])], y_loc, dimension, files_list



