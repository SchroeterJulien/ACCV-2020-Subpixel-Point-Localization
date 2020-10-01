import time
import os
from torch.autograd import Variable
import torch
import random
import numpy as np
import numpy
import networks
from my_args import args
from scipy.misc import imread, imsave
from AverageMeter import *
import shutil
import glob, os
import cv2

import time


downsampling = 16
upsampling = downsampling


torch.backends.cudnn.benchmark = True  # to speed up the

MB_Other_DATA = "./MiddleBurySet/other-data/"
MB_Other_RESULT = "./MiddleBurySet/other-result-author/"
MB_Other_GT = "./MiddleBurySet/other-gt-interp/"
if not os.path.exists(MB_Other_RESULT):
    os.mkdir(MB_Other_RESULT)


args.time_step = 1/upsampling 
args.netName = "DAIN_slowmotion"

model = networks.__dict__[args.netName](channel=args.channels,
                                        filter_size=args.filter_size,
                                        timestep=args.time_step,
                                        training=False)

if args.use_cuda:
    model = model.cuda()

args.SAVED_MODEL = './model_weights/best.pth'
if os.path.exists(args.SAVED_MODEL):
    print("The testing model weight is: " + args.SAVED_MODEL)
    if not args.use_cuda:
        pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
    else:
        pretrained_dict = torch.load(args.SAVED_MODEL)
        # model.load_state_dict(torch.load(args.SAVED_MODEL))

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []
else:
    print("*****************************************************************")
    print("**** We don't load any trained weights **************************")
    print("*****************************************************************")

model = model.eval()  # deploy mode

use_cuda = args.use_cuda
save_which = args.save_which
dtype = args.dtype
unique_id = str(random.randint(0, 100000))
print("The unique id for current testing is: " + str(unique_id))

interp_error = AverageMeter()

#subdir = os.listdir(MB_Other_DATA)
#gen_dir = os.path.join(MB_Other_RESULT, unique_id)


############ Load videos
dir = "GolfDB-master/data/videos_160"
dir_new = dir.replace('videos_160','videos_downsampled_' + str(downsampling) + 'x' + str(upsampling))

if not os.path.exists(dir_new):
    os.mkdir(dir_new)

os.chdir(dir)
list_videos = []
for file in glob.glob("*.mp4"):
    list_videos.append(os.path.join(dir, file))

for video_idx in range(len(list_videos)):

    print(list_videos[video_idx])
    cap = cv2.VideoCapture(list_videos[video_idx])
    fps = cap.get(cv2.CAP_PROP_FPS)
    video = []
    for kk in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, img = cap.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.append(img)
    cap.release()
    video = np.array(video)


    original_length = video.shape[0]
    video_downsampled = video[::downsampling]
    final_upsampled_video = \
        np.zeros([upsampling * (video_downsampled.shape[0]-1)+1,video.shape[1],video.shape[2],video.shape[3]])
    final_upsampled_video[::upsampling] = video_downsampled
    #final_upsampled_video[-(original_length+1)%downsampling:] = video[-(original_length+1)%downsampling:]

    #tot_timer = AverageMeter()
    #proc_timer = AverageMeter()
    end = time.time()
    for frame_idx in range(video_downsampled.shape[0]-1):

        print("----------", video_idx, "//", len(list_videos), "----", frame_idx, "//", video_downsampled.shape[0]-1)
        X0 = torch.from_numpy(np.transpose(video_downsampled[frame_idx,:,:], (2, 0, 1)).astype("float32") / 255.0).type(dtype)
        X1 = torch.from_numpy(np.transpose(video_downsampled[frame_idx+1,:,:], (2, 0, 1)).astype("float32") / 255.0).type(dtype)

        y_ = torch.FloatTensor()

        assert (X0.size(1) == X1.size(1))
        assert (X0.size(2) == X1.size(2))

        intWidth = X0.size(2)
        intHeight = X0.size(1)
        channel = X0.size(0)

        if intWidth != ((intWidth >> 7) << 7):
            intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft = int((intWidth_pad - intWidth) / 2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intWidth_pad = intWidth
            intPaddingLeft = 32
            intPaddingRight = 32

        if intHeight != ((intHeight >> 7) << 7):
            intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intHeight_pad = intHeight
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

        torch.set_grad_enabled(False)
        X0 = Variable(torch.unsqueeze(X0, 0))
        X1 = Variable(torch.unsqueeze(X1, 0))
        X0 = pader(X0)
        X1 = pader(X1)

        if use_cuda:
            X0 = X0.cuda()
            X1 = X1.cuda()
        proc_end = time.time()
        y_s, offset, filter = model(torch.stack((X0, X1), dim=0))
        y_ = y_s[save_which]

        #proc_timer.update(time.time() - proc_end)
        #tot_timer.update(time.time() - end)
        end = time.time()
        print("*****************current image process time \t " + str(time.time() - proc_end) + "s ******************")
        if use_cuda:
            X0 = X0.data.cpu().numpy()
            if not isinstance(y_, list):
                y_ = y_.data.cpu().numpy()
            else:
                y_ = [item.data.cpu().numpy() for item in y_]
            offset = [offset_i.data.cpu().numpy() for offset_i in offset]
            filter = [filter_i.data.cpu().numpy() for filter_i in filter] if filter[0] is not None else None
            X1 = X1.data.cpu().numpy()
        else:
            X0 = X0.data.numpy()
            if not isinstance(y_, list):
                y_ = y_.data.numpy()
            else:
                y_ = [item.data.numpy() for item in y_]
            offset = [offset_i.data.numpy() for offset_i in offset]
            filter = [filter_i.data.numpy() for filter_i in filter]
            X1 = X1.data.numpy()

        # X0 = np.transpose(255.0 * X0.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
        #                           intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))
        y_ = [np.transpose(255.0 * item.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                   intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0)) for item in y_]
        offset = [np.transpose(
            offset_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
            (1, 2, 0)) for offset_i in offset]
        filter = [np.transpose(
            filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
            (1, 2, 0)) for filter_i in filter] if filter is not None else None
        # X1 = np.transpose(255.0 * X1.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
        #                           intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))

        timestep = args.time_step
        numFrames = int(1.0 / timestep) - 1
        time_offsets = [kk * timestep for kk in range(1, 1 + numFrames, 1)]
        # for item, time_offset  in zip(y_,time_offsets):
        #     arguments_strOut = os.path.join(gen_dir, dir, "frame10_i{:.3f}_11.png".format(time_offset))
        #
        #     imsave(arguments_strOut, np.round(item).astype(numpy.uint8))
        #
        # # copy the first and second reference frame
        # shutil.copy(arguments_strFirst, os.path.join(gen_dir, dir,  "frame10_i{:.3f}_11.png".format(0)))
        # shutil.copy(arguments_strSecond, os.path.join(gen_dir, dir,  "frame11_i{:.3f}_11.png".format(1)))

        final_upsampled_video[frame_idx*upsampling+1:(frame_idx+1)*upsampling] = np.array(y_)


    #np.save(list_videos[video_idx].replace('videos_160','videos_downsampled_' + str(downsampling) + 'x').replace('.mp4','.npy'),final_upsampled_video.astype(np.uint8))

    # Write video
    writer = cv2.VideoWriter(list_videos[video_idx].replace('videos_160','videos_downsampled_' + str(downsampling)).replace('.mp4','.mp4'),
                             cv2.VideoWriter_fourcc(*'MP4V'), fps, (160, 160))
    for i in range(final_upsampled_video.shape[0]):
        writer.write(cv2.cvtColor(final_upsampled_video.astype(np.uint8)[i], cv2.COLOR_RGB2BGR))
    writer.release()
    print('saved')
