from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt
import argparse
import librosa
import numpy as np
import os
import pickle
import random
import shutil
import torch

from models.soundNet.soundNet import SoundNet
from models.soundNet.soundNet_outputs import probability_net
from plot_precision_recall import plot_precision_recall_curve

##################################################

# Training command
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_probabilities.py --exp 08_08_22_hard_noise --batch_size 128

######## Arguments to run the code ###############

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp', type=str, required = True, help='name of experiment')
parser.add_argument('--lr', type=float, default=1e-3, help='name of experiment')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--weightdecay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epoch', type=int, default=500, help='The time steps you want to subsample the dataset to')
parser.add_argument('--home_folder', type= str, default='/data/vision/torralba/scratch/fjacob/simplified_transformer/probability_detector/', help='Where are you training your model')
parser.add_argument('--checkpt', type= str, default='23_07_21/1.pth.tar', help='check point')


args = parser.parse_args()

save_folder = 'probability_detector'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

plot_save_folder = './probability_detector/plots'
if not os.path.exists(plot_save_folder):
    os.makedirs(plot_save_folder)

if not os.path.exists('./probability_detector/ckpts'):
    os.makedirs('./probability_detector/ckpts')

if not os.path.exists(os.path.join('./probability_detector/ckpts', args.exp)):
    os.makedirs(os.path.join('./probability_detector/ckpts', args.exp))

if not os.path.exists('./probability_detector/tbs'):
    os.makedirs('./probability_detector/tbs')
    
train_writer = SummaryWriter(os.path.join('./probability_detector/tbs', args.exp , 'train'))
val_writer = SummaryWriter(os.path.join('./probability_detector/tbs', args.exp, 'val'))

def save_checkpoint(state, epoch, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join('probability_detector', 'ckpts', args.exp, filename))
    shutil.copyfile(os.path.join('probability_detector', 'ckpts', args.exp, filename), os.path.join('probability_detector', 'ckpts', args.exp, '{}.pth.tar'.format(str(epoch))))

###### Dataset Loading and Splitting ##########

train_clicks = pickle.load(open('dataset/train_clicks.p',"rb"))
val_clicks = pickle.load(open('dataset/val_clicks.p',"rb"))
test_clicks = pickle.load(open('dataset/test_clicks.p',"rb"))

easy_noise = pickle.load(open('dataset/noise.p',"rb"))
hard_noise_train = pickle.load(open('dataset/train_hard_noise.p',"rb"))
hard_noise_val = pickle.load(open('dataset/val_hard_noise.p',"rb"))
hard_noise_test = pickle.load(open('dataset/test_hard_noise.p',"rb"))

audio_recordings_train = train_clicks + easy_noise[:int(len(train_clicks)/2)] + hard_noise_train[:len(train_clicks)]
audio_recordings_val = val_clicks + easy_noise[int(len(train_clicks)/2):int(len(train_clicks)/2) + int(len(val_clicks)/2)] + hard_noise_val[:len(val_clicks)]
audio_recordings_test = test_clicks + easy_noise[int(len(train_clicks)/2) + int(len(val_clicks)/2):int(len(train_clicks)/2) + int(len(val_clicks)/2) + int(len(test_clicks)/2)] + hard_noise_test[:len(test_clicks)]

######################## Helper functions ######################    
def annotation_inside(window, annotation_array):
    def inside(time_step, window):
        return window[0] <= time_step and window[1] > time_step
    num_clicks = 0
    for time_step in annotation_array:
        if inside(time_step, window):
            num_clicks += 1 
    if num_clicks > 1:
        num_clicks = 1
    return num_clicks

def generate_data(annotation_array, min_timestep, max_timestep, window_size, black_window_size):
    '''
        annotation_array: array of timesteps indicating positives
        min_timestep: the first timestep in the image
        max_timestep: the final timestep in the image
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
        # window_size = 10000
        window_size = 1000
        black_window_size = 40

        audio,sr = librosa.load(self.data_in[idx][0],mono=False)
        label_keys = self.data_in[idx][1]

        if audio.shape[1] >= 10040:
            start_index = 5020 - (window_size + black_window_size)/2 + 1
        else:
            start_index = 0
        end_index = start_index + black_window_size - 1
        curr_green_window, lab = generate_data(label_keys, start_index, end_index, window_size, black_window_size)

        return(audio[:,curr_green_window[0]:curr_green_window[1]],lab)


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


# ########## Training Code ######################
if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    
    seq = SoundNet()
    seq.cuda()
    seq = nn.DataParallel(seq)
    optimizer = optim.Adam(seq.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    valnet = probability_net()
    valnet.cuda()
    valnet = nn.DataParallel(valnet)
    optimizer2 = optim.Adam(valnet.parameters(), lr=args.lr, weight_decay=args.weightdecay)    
    
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = sample_data(audio_recordings_train)
    val_dataset = sample_data(audio_recordings_val)
    test_dataset = sample_data(audio_recordings_test)

    print('Dataset sizes:')
    print(len(train_dataset), len(val_dataset), len(test_dataset))
        
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True, num_workers=20)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=20)    
    
    losses = {'train' : [],
              'val' : []}

    accuracy = {'val' : []}
    
    chkptno = 0
    min_val_loss = None
        
    ########################  TRAINING LOOP ############################
    
    for epoch in range(args.epoch):
        print('\nSTEP: ', epoch)
        seq.train()
        valnet.train()
        avg_train_loss = []
        for i_batch, sample_batched in enumerate(train_dataloader):
            optimizer.zero_grad()
            optimizer2.zero_grad()
            audio = sample_batched[0].type(torch.cuda.FloatTensor) 
            label = sample_batched[1].type(torch.cuda.LongTensor)
            out1 = seq(audio)
            out = valnet(out1)
            loss = criterion(out,label)
            loss.backward()
            optimizer.step()
            optimizer2.step()
            avg_train_loss.append(loss.data.item())
        train_writer.add_scalar('data/loss', np.mean(avg_train_loss) , chkptno)
        avg_train_loss = np.mean(avg_train_loss)        
        print('   Average train loss:', avg_train_loss)
        losses['train'].append(avg_train_loss)
        
        ########################  VALIDATION LOOP ############################
        
        print("\nNow running on val set")
        seq.eval()
        valnet.eval()
        avg_val_loss = []
        for i_batch, sample_batched in enumerate(val_dataloader):
            optimizer.zero_grad()
            optimizer2.zero_grad()
            audio = sample_batched[0].type(torch.cuda.FloatTensor) 
            label = sample_batched[1].type(torch.cuda.LongTensor)
            out1 = seq(audio)
            out = valnet(out1)
            loss = criterion(out,label)
            avg_val_loss.append(loss.data.item())
        val_loss = np.mean(avg_val_loss)
        print('   Average val loss:', val_loss)
        val_writer.add_scalar('data/loss', val_loss, chkptno)

        if chkptno == 0:
            min_val_loss = val_loss

        if val_loss <= min_val_loss:
            min_val_loss = val_loss

            save_checkpoint({
                'epoch': chkptno,
                'state_dict': seq.state_dict(),
                'state_dict_valnet' : valnet.state_dict(),
            }, chkptno)
        
        losses['val'].append(val_loss)
        
        out = out.cpu().data.numpy()
        labels_out = np.argmax(out,axis = 1)
        label = list(label.cpu().data.numpy())
        acc = 0
        for i in range(labels_out.shape[0]):
            if labels_out[i]==label[i]:
                acc +=1
        accuracy['val'].append(acc/labels_out.shape[0])
        print("   Accuracy Val:", acc/labels_out.shape[0])
        val_writer.add_scalar('data/acc', acc/labels_out.shape[0], chkptno)

        chkptno = chkptno+1
        
    ########################  TESTING LOOP ############################

    # Load in best checkpoint model
    checkpoint = torch.load(os.path.join(save_folder, 'ckpts', args.exp, 'checkpoint.pth.tar'))
    seq = SoundNet()
    seq.cuda()
    seq = nn.DataParallel(seq)
    seq.load_state_dict(checkpoint['state_dict'])
    seq.eval()
    valnet = probability_net()
    valnet.cuda()
    valnet = nn.DataParallel(valnet)
    valnet.load_state_dict(checkpoint['state_dict_valnet'])
    valnet.eval()
    
    for i_batch, sample_batched in enumerate(test_dataloader):
        audio = sample_batched[0].type(torch.cuda.FloatTensor) 
        label = sample_batched[1].type(torch.cuda.LongTensor)
        out1 = seq(audio)
        out = valnet(out1)
        # loss = criterion(out,label)
            
    out_numpy = out.cpu().data.numpy()
    labels_out = np.argmax(out_numpy,axis=1)
    label = label.cpu().data.numpy() 
    acc = 0
    for i in range(labels_out.shape[0]):
        if labels_out[i]==label[i]:
            acc +=1
    print('\nTest Accuracy: ', acc/labels_out.shape[0])

    if True:
        train_event_acc = EventAccumulator('./probability_detector/tbs/04_18_22_simplified/train/')
        train_event_acc.Reload()

        val_event_acc = EventAccumulator('./probability_detector/tbs/04_18_22_simplified/val/')
        val_event_acc.Reload()

        _, training_chkptnos, training_loss = zip(*train_event_acc.Scalars('data/loss'))
        _, validation_chkptnos, validation_loss = zip(*val_event_acc.Scalars('data/loss'))
        _, validation_chkptnos, validation_accuracy = zip(*val_event_acc.Scalars('data/acc'))

        losses['train'] = training_loss
        losses['val'] = validation_loss
        accuracy['val'] = validation_accuracy

    plot_losses_and_accuracy(losses, accuracy, plot_save_folder)
    save_path = os.path.join(save_folder,'plots', args.exp, 'precision_recall.png')
    out_probabilities = nn.functional.softmax(out, dim=1)
    out_probabilities = out_probabilities.cpu().data.numpy()
    out_probabilities = out_probabilities[:,1]  # .reshape((out_probabilities.shape[1]))
    plot_precision_recall_curve(label, out_probabilities, save_path)
