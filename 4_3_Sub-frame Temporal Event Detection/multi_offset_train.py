from dataloader import GolfDB, Normalize, ToTensor
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt

from multi_offset_model import EventDetector


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import SmoothedLosses as sloss
from Math.NadarayaWatson import *

# Get the arguments from the command-line except the filename
import sys
argv = sys.argv[1:]
if len(argv)>0:
    step = int(argv[0])
    print('>>> step: ' + str(step))
else:
    step = 7


if len(argv)>1:
    run_idx = int(argv[1])
else:
    run_idx = 10

if len(argv)>2:
    split = int(argv[2])
else:
    split = 1


config = {'name': 'multi_v4', 'Adam': True, 'iterations': 10000, 'show_frequency':500, 'alpha':0.01,
          'learning_rate':0.0002, 'start_converging': 1000}

version_name = config['name'] + '_' + str(step) + '_' + str(run_idx) + '_' + str(split)
print(version_name)

loss_window = {'loss': np.zeros([25]), 'soft': np.zeros([25]), 'count': np.zeros([25])}
list_loss = {'loss':[], 'soft':[], 'count':[]}


if __name__ == '__main__':

    # training configuration
    iterations = config['iterations']
    it_save = 10000  # save model every 100 iterations
    n_cpu = 8
    seq_length = 65
    bs = 22  # batch size
    k = 10  # frozen layers

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False,
                          step=step)
    freeze_layers(k, model)
    model.train()
    model.to(device)

    dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=True,
                     step=step)

    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)


    weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    if config['Adam']:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate']) #todo(default):0.001
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate']) #todo(default):0.001 #todo(default):0.001

    losses = AverageMeter()

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0
    while i < iterations:
        for sample in data_loader:
            smoothing_labda = max(8 - max((i - 2000) / 1500, 0), 2)

            images, labels = sample['images'][:,::step,:,:,:].to(device), sample['labels']
            preds = model(images)

            # Transform labels
            location_true = torch.zeros(torch.sum((labels!=8).int()), preds.shape[2]+1)
            counter = 0
            for kk in range(labels.shape[0]):
                for location in torch.where(labels[kk]!=8)[0]:
                    location_true[counter, 0] = kk
                    location_true[counter, 1]  = location
                    location_true[counter, labels[kk][location]+2]  = 1
                    counter+=1

            location_true = location_true.to(device)

            # Compute loss
            loss_smooth, loss_count = sloss.compute_loss_smoothed(preds, location_true, labels.to(device), smoothing_labda)
            loss = loss_smooth + min(max(i-config['start_converging'],0)/8000,1) * config['alpha'] * loss_count

            loss_window['loss'][1:] = loss_window['loss'][:-1]
            loss_window['loss'][0] = loss.cpu().data.numpy()
            list_loss['loss'].append(np.median(loss_window['loss'][loss_window['loss'] != 0]))

            loss_window['soft'][1:] = loss_window['soft'][:-1]
            loss_window['soft'][0] = loss_smooth.cpu().data.numpy()
            list_loss['soft'].append(np.median(loss_window['soft'][loss_window['soft'] != 0]))

            loss_window['count'][1:] = loss_window['count'][:-1]
            loss_window['count'][0] = config['alpha'] * loss_count.cpu().data.numpy()
            list_loss['count'].append(np.median(loss_window['count'][loss_window['count'] != 0]))


            if i % config['show_frequency'] == 0:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.plot(np.log(list_loss['loss']), 'k', alpha=0.5, linewidth=1)
                plt.plot(1 * np.arange(1, 1 + len(list_loss['loss'])),
                         GaussKernel(np.arange(1, 1 + len(list_loss['loss'])),
                                     np.arange(1, 1 + len(list_loss['loss'])),
                                     np.log(np.array(list_loss['loss'])), config['show_frequency']),
                         'k', linewidth=2)

                plt.plot(np.log(list_loss['soft']), 'b', alpha=0.5, linewidth=1)
                plt.plot(1 * np.arange(1, 1 + len(list_loss['soft'])),
                         GaussKernel(np.arange(1, 1 + len(list_loss['soft'])),
                                     np.arange(1, 1 + len(list_loss['soft'])),
                                     np.log(np.array(list_loss['soft'])), config['show_frequency']),
                         'b', linewidth=2)

                plt.plot(np.log(np.array(list_loss['count'])), 'orange', alpha=0.5, linewidth=1)
                plt.plot(1 * np.arange(1, 1 + len(list_loss['count'])),
                         GaussKernel(np.arange(1, 1 + len(list_loss['count'])),
                                     np.arange(1, 1 + len(list_loss['count'])),
                                     np.log(np.array(list_loss['count'])), config['show_frequency']),
                         'orange', linewidth=2)

                plt.ylim([np.log(min(np.min(list_loss['loss']), np.min(list_loss['soft']), np.min(list_loss['count']))), np.log(list_loss['loss'][min(500,max(0,i-100))])])

                plt.subplot(1, 2, 2)
                plt.hist(preds[:,:,1:].reshape(-1).cpu().data.numpy())
                plt.ylim([0,50])

                plt.savefig('loss_' + version_name + '.png')
                plt.close('all')


            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            if i % 10 == 0:
                print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            i += 1
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/multi_{}_{}.pth.tar'.format(version_name, i))
            if i == iterations:
                break





