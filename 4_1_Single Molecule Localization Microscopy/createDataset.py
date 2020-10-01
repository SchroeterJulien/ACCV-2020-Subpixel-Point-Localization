import numpy as np
import os
from skimage import io

# Name of set
mat_filename = 'TrainingSet'

# From https://github.com/EliasNehme/Deep-STORM/tree/master/demo%201%20-%20Simulated%20Microtubules
datapath = 'benchmark_data/demo 1 - Simulated Microtubules'
tiff_filename = 'benchmark_data/ArtificialDataset_demo1.tif'
csv_filename = 'benchmark_data/positions_demo1.csv'

# Settings as in benchmark
camera_pixelsize = 100
patch_size = 26
margin_context = 0
num_patches = 500

# Read the artificial acquisition stack
im = io.imread(os.path.join(datapath, tiff_filename))

# Parse positions
positions = np.genfromtxt(os.path.join(datapath, csv_filename), delimiter=',')[1:, :]

# Create dataset by cropping radom sub-patches of full image
count_img = 0
data_image = np.zeros([im.shape[0] * 500, patch_size + 2 * margin_context, patch_size + 2 * margin_context])
data_positions = np.zeros([im.shape[0] * 500, 100, 3])
for idx_image in range(im.shape[0]):
    for pp in range(500):

        # Process image
        image_position = positions[positions[:, 1] == (idx_image + 1), 2:4] / camera_pixelsize

        xx_start, yy_start = np.random.randint(0, im.shape[1] - patch_size - 2 * margin_context), \
                             np.random.randint(0, im.shape[2] - patch_size - 2 * margin_context)
        img_patch = im[idx_image][xx_start:xx_start + patch_size + 2 * margin_context,
                    yy_start:yy_start + patch_size + 2 * margin_context]

        data_image[count_img] = img_patch

        position_patch = np.copy(image_position)
        position_patch = position_patch[np.logical_and(position_patch[:, 0] >= yy_start + margin_context,
                                                       position_patch[:, 0] <= yy_start + patch_size + margin_context)]
        position_patch = position_patch[np.logical_and(position_patch[:, 1] >= xx_start + margin_context,
                                                       position_patch[:, 1] <= xx_start + patch_size + margin_context)]
        position_patch[:, 0] -= yy_start
        position_patch[:, 1] -= xx_start

        # Process Labels
        data_positions[count_img, :position_patch.shape[0], 0] = position_patch[:, 0]
        data_positions[count_img, :position_patch.shape[0], 1] = position_patch[:, 1]
        data_positions[count_img, :position_patch.shape[0], 2] = 1

        count_img += 1


# Save
if not os.path.exists('data'):
    os.makedirs('data')

np.save('data/our_data_26.npy', data_image)
np.save('data/our_labels_26.npy', data_positions)
