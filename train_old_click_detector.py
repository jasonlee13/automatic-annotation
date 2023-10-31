from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import argparse
# import librosa
import numpy as np
import os
import pickle
import random
import shutil
import torch
import scipy.io.wavfile

##################################################

# OLD VERSION: 03_28_22_hard_noise

# Training Command
# CUDA_VISIBLE_DEVICES=0,1,2,3 python new_click_detector.py --exp 07_29_22_train_for_comparison_new_dataset --batch_size 128 --black_window_size 40 --window_size 1000

######## Arguments to run the code ###############

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp', type=str, required = True, help='name of experiment')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--weightdecay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epoch', type=int, default=500, help='Number of epochs to run')
parser.add_argument('--home_folder', type= str, default='/data/vision/torralba/scratch/fjacob/models/click_detector/', help='Where are you training your model')
parser.add_argument('--test', type=bool, default=False, help='Testing time')

parser.add_argument('--min_timestep', type=int, default=0, help='min_timestep')
parser.add_argument('--max_timestep', type=int, default=120000, help='max_timestep')
parser.add_argument('--window_size', type=int, default=2000, help='window_size')
parser.add_argument('--black_window_size', type=int, default=150, help='black_window_size')

args = parser.parse_args()

plot_save_folder = './plots'
if not os.path.exists(plot_save_folder):
    os.makedirs(plot_save_folder)

if not os.path.exists('./ckpts'):
    os.makedirs('./ckpts')

if not os.path.exists(os.path.join('./ckpts', args.exp)):
    os.makedirs(os.path.join('./ckpts', args.exp))

if not os.path.exists('./tbs'):
    os.makedirs('./tbs')
    
train_writer = SummaryWriter(os.path.join('./tbs', args.exp , 'train'))
val_writer = SummaryWriter(os.path.join('./tbs', args.exp, 'val'))

def save_checkpoint(state, epoch, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(args.home_folder, 'ckpts', args.exp, filename))
    shutil.copyfile(os.path.join(args.home_folder, 'ckpts', args.exp, filename), os.path.join(args.home_folder, 'ckpts', args.exp, '{}.pth.tar'.format(str(epoch))))


###### Dataset Loading and Splitting##########

# TODO!!!!! Make this so I am training on the new datasets used by all other models.

data_directory = '/data/vision/torralba/scratch/fjacob/models/shuffled_click_detector_dataset.p'
total_data = pickle.load(open(data_directory,"rb"))

print('Number of click examples: {}'.format(len(total_data)))

audio_recordings_train = total_data[:int(0.8*len(total_data))]
audio_recordings_val = total_data[int(0.8*len(total_data)):int(0.9*len(total_data))]
audio_recordings_test = total_data[int(0.9*len(total_data)):]

noise_directory = '/data/vision/torralba/scratch/fjacob/models/shuffled_click_detector_noise.p'
noise_data = pickle.load(open(noise_directory,"rb"))

print('Number of noise examples: {}'.format(len(noise_data)))

audio_recordings_train += noise_data[:int(0.8*len(noise_data))]
audio_recordings_val += noise_data[int(0.8*len(noise_data)):int(0.9*len(noise_data))]
audio_recordings_test += noise_data[int(0.9*len(noise_data)):]

hard_noise_directory = '/data/vision/torralba/scratch/fjacob/models/shuffled_click_detector_hard_noise.p'
hard_noise_data = pickle.load(open(noise_directory,"rb"))

print('Number of hard noise examples: {}'.format(len(hard_noise_data)))

audio_recordings_train += hard_noise_data[:int(0.8*len(hard_noise_data))]
audio_recordings_val += hard_noise_data[int(0.8*len(hard_noise_data)):int(0.9*len(hard_noise_data))]
audio_recordings_test += hard_noise_data[int(0.9*len(hard_noise_data)):]


######################## Helper functions ######################   
def annotation_inside(window, annotation_array):
    #TODO: ensure only one annotation lies inside the window
    def inside(time_step, window):
        return window[0] <= time_step and window[1] >= time_step
    for time_step in annotation_array:
        if inside(time_step, window):
            return time_step - window[0] + 1 
    return 0

def generate_data(annotation_array, min_timestep, max_timestep, window_size, black_window_size):
    '''
        annotation_array: array of timesteps indicating positives
        min_timestep: the first timestep in the image
        max_timestep: the final timestep in the image
        time_scale: the granularity of the time scale
        window_size: size of the "black" window
        return:
            a collection of data points generated following the natural distribution
    '''

    annotation_array.sort()
    window_st = random.randint(min_timestep, max_timestep)
    curr_green_window = [window_st, window_st + window_size]
    black_window_st = window_st + (window_size - black_window_size)/2
    black_window_en = black_window_st + black_window_size

    label = annotation_inside((black_window_st, black_window_en), annotation_array)
    return (curr_green_window, label)


class sample_data(Dataset):
    def __init__(self, data_in): 
        self.data_in = data_in

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, idx):
        sr, audio = scipy.io.wavfile.read(self.data_in[idx][0])
        audio = audio.T
        audio = audio / 32768.0

        label_keys = self.data_in[idx][1]
        
        prob_noise = random.uniform(0, 1)

        # Choose data randomly from in audio or just close to center

        # Prob_noise used to be 0.75, trying lower to see...
        if prob_noise>0.75:
            curr_green_window, lab = generate_data(label_keys, args.min_timestep, args.max_timestep-args.window_size, args.window_size, args.black_window_size)  
        else:
            # curr_green_window, lab = generate_data(label_keys, int(sr/2-args.window_size/2)- args.black_window_size, int(sr/2-args.window_size/2), args.window_size, args.black_window_size)
            curr_green_window, lab = generate_data(label_keys, int(sr/2-args.window_size/2 - args.black_window_size/2), int(sr/2-args.window_size/2 + args.black_window_size/2), args.window_size, args.black_window_size)

        return(audio[:,curr_green_window[0]:curr_green_window[1]],lab)
            
###### Model #################################

class SoundNet(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1),
                               padding=(32, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1),
                               padding=(16, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1),
                               padding=(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1),
                               padding=(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)

        self.conv8_objs = nn.Conv2d(1024, 1000, kernel_size=(8, 1),
                                    stride=(2, 1))
        self.conv8_scns = nn.Conv2d(1024, 401, kernel_size=(8, 1),
                                    stride=(2, 1))

    def forward(self, waveform):
        x = self.conv1(waveform.unsqueeze(1).permute(0,1,3,2))
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        x = x.reshape(x.shape[0],-1)
        return x

    
class value_net(nn.Module):
    def __init__(self, symmetric=True):
        super(value_net, self).__init__()
        self.linear = nn.Linear(768, args.black_window_size+2)  # Window size: 768 for 2000, 1280 for 4000, 1792 for 6000, 2304 for 8000, 2816 for 10000
    
    def forward(self, input_audio):
        print(input_audio.shape)
        output = self.linear(input_audio)
        return output


def plot_losses_and_accuracy(losses, accuracy, save_folder):
    if not os.path.exists(os.path.join(save_folder, args.exp)):
        os.mkdir(os.path.join(save_folder, args.exp))
    # Training and validation loss
    fig, axs = plt.subplots(2, 1, figsize=(10,20))
    axs[0].plot([i for i in range(len(losses['train']))], losses['train'])
    axs[0].set_title('Training Loss')
    axs[0].set_ylabel('Loss')
    axs[1].plot([i for i in range(len(losses['val']))], losses['val'])
    axs[1].set_title('Validation Loss')
    axs[1].set_xlabel('Training Epochs')
    axs[1].set_ylabel('Loss')
    plt.savefig(os.path.join(save_folder, args.exp, 'loss.png'))
    plt.close()

    # Validation accuracy
    fig, axs = plt.subplots(figsize=(10,10))
    axs.plot([i for i in range(len(accuracy['val']))], accuracy['val'])
    axs.set_title('Validation Accuracy')
    axs.set_ylabel('Accuracy')
    axs.set_xlabel('Training Epochs')
    plt.savefig(os.path.join(save_folder, args.exp, 'accuracy.png'))
    plt.close()

################################ Training Code ############################################

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    
    seq = SoundNet()
    valnet = value_net()
    valnet.cuda()
    valnet = nn.DataParallel(valnet)
    optimizer2 = optim.Adam(valnet.parameters(), lr=args.lr, weight_decay=args.weightdecay)  

    seq.cuda()
    seq = nn.DataParallel(seq)  
    
    optimizer = optim.Adam(seq.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = sample_data(audio_recordings_train)
    val_dataset = sample_data(audio_recordings_val)
    test_dataset = sample_data(audio_recordings_test)

    print('Train data size: {}, Val data size: {}, Test data size: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=not(args.test), num_workers=20)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset),
                            shuffle=True, num_workers=20)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset),
                            shuffle=False, num_workers=20)
    
    losses = {'train' : [],
              'val' : []}

    accuracy = {'val' : []}

    chkptno = 0
    min_val_loss = None

    for epoch in range(args.epoch):
        print('STEP: ', epoch)
        seq.train()
        valnet.train()
        avg_train_loss = []
        non_zero_labels = []
        for i_batch, sample_batched in enumerate(train_dataloader):
            audio = sample_batched[0].type(torch.cuda.FloatTensor)
            label = sample_batched[1].type(torch.cuda.LongTensor)
            out = seq(audio)
            out = valnet(out)
            loss = criterion(out,label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()
            avg_train_loss.append(loss.data.item())

            label = label.cpu().data.numpy()
            for i in range(len(label)):
                if label[i] != 0:
                    non_zero_labels.append(label[i])

        print("Average Train Loss: ",np.mean(avg_train_loss))
        print('Min label: {}, Max label: {}, Avg label: {}'.format(min(non_zero_labels), max(non_zero_labels), sum(non_zero_labels)/len(non_zero_labels)))
        train_writer.add_scalar('data/loss', np.mean(avg_train_loss) , chkptno)
        losses['train'].append(np.mean(avg_train_loss))
        
        #########################################  Validation Loop #########################
        print("Now running on val set")
        seq.eval()
        valnet.eval()
        avg_val_loss = []
        for i_batch, sample_batched in enumerate(val_dataloader):
            audio = sample_batched[0].type(torch.cuda.FloatTensor)
            label = sample_batched[1].type(torch.cuda.LongTensor)
            out = seq(audio)
            out = valnet(out)
            loss = criterion(out,label)
            avg_val_loss.append(loss.cpu().data.item())
        val_loss = np.mean(avg_val_loss)
        print('Average val loss:', val_loss)
        val_writer.add_scalar('data/loss', val_loss, chkptno)
        losses['val'].append(val_loss)

        if chkptno == 0:
            min_val_loss = val_loss

        if val_loss <= min_val_loss:
            min_val_loss = val_loss

            save_checkpoint({
                'epoch': chkptno,
                'state_dict': seq.state_dict(),
                'state_dict_valnet' : valnet.state_dict(),
            }, chkptno)
        
        # Compute accuracy
        out = out.cpu().data.numpy()
        labels_out = np.argmax(out,axis = 1)
        label = label.cpu().data.numpy() 
        acc = 0
        for i in range(labels_out.shape[0]):
            if labels_out[i]==label[i]:
                acc +=1
        print("Accuracy Val :", acc/labels_out.shape[0])
        accuracy['val'].append(acc/labels_out.shape[0])

        acc_click = 0
        total_click = 0
        for i in range(labels_out.shape[0]):
            if label[i] != 0:
                total_click += 1
                if labels_out[i]==label[i]:
                    acc_click +=1
        print("Click Accuracy Val :", acc_click / total_click)
        
        chkptno = chkptno+1

    #################################### Test Loop #####################################

    # Load in best checkpoint model
    checkpoint = torch.load(os.path.join(args.home_folder, 'ckpts', args.exp, 'checkpoint.pth.tar'))
    seq = SoundNet()
    seq.cuda()
    seq = nn.DataParallel(seq)
    seq.load_state_dict(checkpoint['state_dict'])
    seq.eval()
    valnet = value_net()
    valnet.cuda()
    valnet = nn.DataParallel(valnet)
    valnet.load_state_dict(checkpoint['state_dict_valnet'])
    valnet.eval()
    
    avg_test_loss = []
    for i_batch, sample_batched in enumerate(test_dataloader):
        audio = sample_batched[0].type(torch.cuda.FloatTensor)
        label = sample_batched[1].type(torch.cuda.LongTensor)
        out = seq(audio)
        out = valnet(out)
        loss = criterion(out,label)
        avg_test_loss.append(loss.cpu().data.item())
    test_loss = np.mean(avg_test_loss)
    # print('Average test loss:', test_loss)
    
    # Compute accuracy
    out = out.cpu().data.numpy()
    labels_out = np.argmax(out,axis = 1)
    label = label.cpu().data.numpy() 
    acc = 0
    for i in range(labels_out.shape[0]):
        if labels_out[i]==label[i]:
            acc +=1
    print("Accuracy Test :", acc/labels_out.shape[0])

    acc_click = 0
    total_click = 0
    for i in range(labels_out.shape[0]):
        if label[i] != 0:
            total_click += 1
            if labels_out[i]==label[i]:
                acc_click +=1
    print("Click Accuracy Test :", acc_click / total_click)

    plot_losses_and_accuracy(losses, accuracy, plot_save_folder)