import os.path as osp
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True, step=1):
        self.df = pd.read_pickle(data_file)#[:200]
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train
        self.step = step

        if self.train and False:
            self.max_shape = 600
            self.videos_full = np.zeros([len(self.df), self.max_shape, 160,160,3], dtype=np.uint8)
            self.labels_full = []

            print('Load images to memory...')
            for idx in range(len(self.df)):
                # print(idx)
                a = self.df.loc[idx, :]  # annotation info
                events = a['events']
                events -= events[0]  # now frame #s correspond to frames in preprocessed video clips

                images, labels = [], []
                cap = cv2.VideoCapture(osp.join(self.vid_dir, '{}.mp4'.format(a['id'])))
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                pos = 0
                idx_frame = 0
                while len(images) < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                    ret, img = cap.read()
                    if ret:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.videos_full[idx, idx_frame,:,:,:] = img
                        if pos in events[1:-1]:
                            labels.append(np.where(events[1:-1] == pos)[0][0])
                        else:
                            labels.append(8)
                        pos += 1
                        idx_frame+=1
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        pos = 0
                cap.release()
                for kk in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.max_shape):
                    self.videos_full[idx, kk, :, :, :] = self.videos_full[idx, idx_frame-1, :, :, :]

                # images += self.seq_length * [images[-1]]  # copy last frame
                labels += self.seq_length * [labels[-1]]  # copy last labels

                # self.videos_full.append(images)
                self.labels_full.append(labels)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        if not hasattr(self, 'videos_full'):

            a = self.df.loc[idx, :]  # annotation info
            events = a['events']
            events -= events[0]  # now frame #s correspond to frames in preprocessed video clips

            images, labels = [], []
            cap = cv2.VideoCapture(osp.join(self.vid_dir, '{}.mp4'.format(a['id'])))
            #print(osp.join(self.vid_dir, '{}.mp4'.format(a['id'])))

            if self.train:
                # random starting position, sample 'seq_length' frames
                start_frame =  self.step * (np.random.randint(events[-1]  + 1) // self.step) # info: downsamples the video

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                pos = start_frame
                while len(images) < self.seq_length:
                    ret, img = cap.read()
                    if ret:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        if pos in events[1:-1]:
                            labels.append(np.where(events[1:-1] == pos)[0][0])
                        else:
                            labels.append(8)
                        pos += 1
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        pos = 0
                cap.release()
            else:
                # full clip
                for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                    _, img = cap.read()
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    if pos in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos)[0][0])
                    else:
                        labels.append(8)
                cap.release()
        else:

            a = self.df.loc[idx, :]  # annotation info
            events = a['events']
            events -= events[0]

            start_frame = self.step * (np.random.randint(events[-1] + 1) // self.step)  # info: downsamples the video

            images = self.videos_full[idx, start_frame:start_frame + self.seq_length]
            labels = self.labels_full[idx][start_frame:start_frame + self.seq_length]
            # print(np.asarray(self.videos_full[idx][:]).shape, np.asarray(self.labels_full[idx]).shape)
            # print(np.asarray(images).shape, np.asarray(labels).shape)

        sample = {'images':np.asarray(images), 'labels':np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.),
                'labels': torch.from_numpy(labels).long()}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}


if __name__ == '__main__':

    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std (RGB)

    dataset = GolfDB(data_file='data/train_split_1.pkl',
                     vid_dir='data/videos_160/',
                     seq_length=64,
                     transform=transforms.Compose([ToTensor(), norm]),
                     train=False)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 8)[0]
        print('{} events: {}'.format(len(events), events))




    





       

