import cv2
import numpy as np
import glob
import os

dwn_rate = [1/2, 1/4, 1/6]


# Mesa
files_list = glob.glob('Checkerboard/benchmarkData/Mesa/*.png')
for dwn_index in range(len(dwn_rate)):
    try:
        os.makedirs('Checkerboard/benchmarkData/Mesa/'.replace('/Mesa/', '/Mesa_dwn_{0}/'.format(dwn_index)))
    except:
        pass
    for kk in range(len(files_list)):
        print(kk, len(files_list))
        img = cv2.imread(files_list[kk],0)
        def scalingImage(checkerboard, real_corners=None, scaling=None):
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
            if real_corners is not None:
                warped_points = cv2.perspectiveTransform(real_corners, M)
            else:
                warped_points = cv2.perspectiveTransform(original, M)

            return warped, warped_points


        small = cv2.GaussianBlur(img.astype(np.uint8), (11, 11), 2/dwn_rate[dwn_index])#1/dwn_rate[dwn_index])  # blur
        small, pp = scalingImage(small,scaling=(dwn_rate[dwn_index], dwn_rate[dwn_index])) # downscaling

        cv2.imwrite(files_list[kk].replace('/Mesa/', '/Mesa_dwn_{0}/'.format(dwn_index)).replace('.png','.JPG'), small)




# uEye
files_list = glob.glob('Checkerboard/benchmarkData/uEye/*.png')
for dwn_index in range(len(dwn_rate)):
    try:
        os.makedirs('Checkerboard/benchmarkData/uEye/'.replace('/uEye/', '/uEye_dwn_{0}/'.format(dwn_index)))
    except:
        pass
    for kk in range(len(files_list)):
        print(kk, len(files_list))
        img = cv2.imread(files_list[kk],0)
        def scalingImage(checkerboard, real_corners=None, scaling=None):
            if scaling is None:
                scaling = np.random.uniform(0.5, 1.2, size=2)
            if not hasattr(scaling, "__len__"):
                scaling = (scaling, scaling)
            # warped = cv2.resize(warped, (0, 0), fx=scaling[0], fy=scaling[1], interpolation=cv2.INTER_LINEAR)
            # points *= np.array([scaling[0], scaling[1]])[np.newaxis, np.newaxis, :]

            original = np.array([[100, 100], [200, 100], [100, 200], [200, 200]], np.float32)[:, np.newaxis, :]
            projected = np.copy(original)
            projected[:, :, 0] *= scaling[0]
            projected[:, :, 1] *= scaling[1]

            M = cv2.getPerspectiveTransform(original.astype(np.float32), projected.astype(np.float32))
            warped = cv2.warpPerspective(checkerboard, M, (
                int(np.ceil(scaling[0] * checkerboard.shape[1])), int(np.ceil(scaling[1] * checkerboard.shape[0]))))
            if real_corners is not None:
                warped_points = cv2.perspectiveTransform(real_corners, M)
            else:
                warped_points = cv2.perspectiveTransform(original, M)

            return warped, warped_points


        small = cv2.GaussianBlur(img.astype(np.uint8), (11, 11), 2/dwn_rate[dwn_index])#1/dwn_rate[dwn_index])  # blur
        small, pp = scalingImage(small,scaling=(dwn_rate[dwn_index], dwn_rate[dwn_index])) # downscaling

        cv2.imwrite(files_list[kk].replace('/uEye/', '/uEye_dwn_{0}/'.format(dwn_index)).replace('.png','.png'), small)


# GoPro
files_list = glob.glob('Checkerboard/benchmarkData/GoPro/*.JPG')
for dwn_index in range(len(dwn_rate)):
    try:
        os.makedirs('Checkerboard/benchmarkData/GoPro/'.replace('/GoPro/', '/GoPro_dwn_{0}/'.format(dwn_index)))
    except:
        pass
    for kk in range(len(files_list)):
        print(kk, len(files_list))
        img = cv2.imread(files_list[kk],0)

        def scalingImage(checkerboard, real_corners=None, scaling=None):
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
            if real_corners is not None:
                warped_points = cv2.perspectiveTransform(real_corners, M)
            else:
                warped_points = cv2.perspectiveTransform(original, M)

            return warped, warped_points


        small = cv2.GaussianBlur(img.astype(np.uint8), (11, 11), 1/dwn_rate[dwn_index])  # blur
        small, pp = scalingImage(small,scaling=(dwn_rate[dwn_index], dwn_rate[dwn_index])) # downscaling

        cv2.imwrite(files_list[kk].replace('/GoPro/', '/GoPro_dwn_{0}/'.format(dwn_index)).replace('.JPG','.JPG'), small)
