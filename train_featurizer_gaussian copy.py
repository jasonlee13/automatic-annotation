##################################################
# Training command
##################################################
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
# python -m train.train_featurizer_gaussian --exp test_gaussian --lr 1e-3 --num_per_convo 128 --batch_size 512 --embedding_size 1024
# nohup python -m train.train_featurizer.py --exp n_dim_40 --lr 1e-3 --num_per_convo 128 --batch_size 512 --embedding_size 40 > n_dim_40.out &
##################################################

from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import pandas as pd
import argparse
import librosa
import numpy as np
import os
import pickle
import random
import shutil
import torch
import math
import time

from models.soundNet.soundNet import SoundNet, SoundNet2
from models.soundNet.soundNet_outputs import probability_net, confidence_net, click_time_net, whale_embedding_net, whale_embedding_net1
from models.soundNet.mlp_output import mlp
from eval.plot_precision_recall import plot_precision_recall_curve
from losses.supervised_contrastive_loss import SupConLoss

import wandb

######## Arguments to run the code ###############

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "11"

assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == torch.cuda.device_count(), "Number of cuda devices not equal to number of visible devices"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp', type=str, required = True, help='Name of the experiment')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_per_convo', type=int, default=8, help='Number of clicks per conversation for contrastive loss')
parser.add_argument('--embedding_size', type=int, default=20, help='Size of the whale embedding')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--weightdecay', type=float, default=1e-4, help='Weight Decay')
parser.add_argument('--epoch', type=int, default=500, help='Number of epochs')
parser.add_argument('--home_folder', type= str, default='/data/vision/torralba/scratch/fjacob/simplified_transformer/featurizer/', help='Where you are training')
parser.add_argument('--checkpt', type= str, default= time.strftime("%d_%m_%y_%H_%M") + '.pth.tar', help='Check Point')

args = parser.parse_args()

save_folder = 'featurizer'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

plot_save_folder = './featurizer/plots'
if not os.path.exists(plot_save_folder):
    os.makedirs(plot_save_folder)

if not os.path.exists('./featurizer/ckpts'):
    os.makedirs('./featurizer/ckpts')

if not os.path.exists(os.path.join('./featurizer/ckpts', args.exp)):
    os.makedirs(os.path.join('./featurizer/ckpts', args.exp))

if not os.path.exists('./featurizer/tbs'):
    os.makedirs('./featurizer/tbs')
    
train_writer = SummaryWriter(os.path.join('./featurizer/tbs', args.exp , 'train'))
val_writer = SummaryWriter(os.path.join('./featurizer/tbs', args.exp, 'val'))

def save_checkpoint(state, epoch, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join('featurizer', 'ckpts', args.exp, filename))
    shutil.copyfile(os.path.join('featurizer', 'ckpts', args.exp, filename), os.path.join('featurizer', 'ckpts', args.exp, '{}.pth.tar'.format(str(epoch))))



remake_dataset = False
if remake_dataset:
    new_conversations = pickle.load(open('dataset/new_conversations.p', 'rb'))
    print(len(new_conversations))
    total_number_of_clicks = sum([len(convo) for convo in new_conversations])

    train_conversations = []
    num_train_clicks = 0
    while num_train_clicks < total_number_of_clicks * 0.8:
        next_convo = new_conversations.pop()
        train_conversations.append(next_convo)
        num_train_clicks += len(next_convo)

    val_conversations = []
    num_val_clicks = 0
    while num_val_clicks < total_number_of_clicks * 0.1:
        next_convo = new_conversations.pop()
        val_conversations.append(next_convo)
        num_val_clicks += len(next_convo)

    test_conversations = []
    while len(new_conversations) > 0:
        next_convo = new_conversations.pop()
        test_conversations.append(next_convo)

    print('Saving datasets...')
    with open('dataset/new_conversations.p', 'wb') as pickle_file:
        pickle.dump(new_conversations, pickle_file)
    with open('dataset/train_conversations.p', 'wb') as pickle_file:
        pickle.dump(train_conversations, pickle_file)
    with open('dataset/val_conversations.p', 'wb') as pickle_file:
        pickle.dump(val_conversations, pickle_file)
    with open('dataset/test_conversations.p', 'wb') as pickle_file:
        pickle.dump(test_conversations, pickle_file)




# train_conversations = pickle.load(open('dataset/train_conversations.p',"rb"))
# train_noise = pickle.load(open('dataset/train_noise.p',"rb"))
# val_conversations = pickle.load(open('dataset/val_conversations.p',"rb"))
# val_noise = pickle.load(open('dataset/val_noise.p',"rb"))
# test_conversations = pickle.load(open('dataset/test_conversations.p',"rb"))
# test_noise = pickle.load(open('dataset/test_noise.p',"rb"))

# hard_noise_train = pickle.load(open('dataset/train_hard_noise.p',"rb"))
# hard_noise_val = pickle.load(open('dataset/val_hard_noise.p',"rb"))
# hard_noise_test = pickle.load(open('dataset/test_hard_noise.p',"rb"))


data_path = "/data/vision/torralba/scratch/fjacob/simplified_transformer/dataset"

train_conversations = pickle.load(open(os.path.join(data_path, 'train_conversations.p'), "rb"))
train_noise = pickle.load(open(os.path.join(data_path, 'train_noise.p'), "rb"))
val_conversations = pickle.load(open(os.path.join(data_path, 'val_conversations.p'), "rb"))
val_noise = pickle.load(open(os.path.join(data_path, 'val_noise.p'), "rb"))
test_conversations = pickle.load(open(os.path.join(data_path, 'test_conversations.p'), "rb"))
test_noise = pickle.load(open(os.path.join(data_path, 'test_noise.p'), "rb"))

hard_noise_train = pickle.load(open(os.path.join(data_path, 'train_hard_noise.p'), "rb"))
hard_noise_val = pickle.load(open(os.path.join(data_path, 'val_hard_noise.p'), "rb"))
hard_noise_test = pickle.load(open(os.path.join(data_path, 'test_hard_noise.p'), "rb"))



wandb.init(project="whale-featurizer",
             name=args.exp + '_' + time.strftime("%d_%m_%y_%H_%M_%S"),
            config=args)


######################## Helper functions ###################### 
def order_audio_windows(conversations, noise, hard_noise, clicks_per_convo, num_samples=None):
    if num_samples is None:
        num_samples = len(conversations) + len(noise)
    noise_copy = noise.copy()
    hard_noise_copy = hard_noise.copy()
    ordered_audio_windows = []
    still_ordering = True
    while still_ordering and len(ordered_audio_windows) < num_samples:
        still_ordering = False
        new_conversations = []
        for i in range(len(conversations)):
            if len(conversations[i]) >= clicks_per_convo:
                max_whale_number = max([whale[2][0] for whale in conversations[i]])
                sorted_conversations = dict()
                valid_indices = []
                double_valid = []
                for j in range(1, max_whale_number+1):
                    whale_j_clicks = [whale for whale in conversations[i] if whale[2][0]==j]
                    if len(whale_j_clicks) >= clicks_per_convo/2:
                        sorted_conversations[j] = whale_j_clicks
                        valid_indices.append(j)
                    if len(whale_j_clicks) >= clicks_per_convo:
                        double_valid.append(j)
                if len(valid_indices) >= 2 or len(double_valid) >= 1:
                    still_ordering = True
                    whale_1 = valid_indices[random.randint(0, len(valid_indices)-1)]
                    # if len(sorted_conversations[whale_1]) < clicks_per_convo:
                    #     valid_indices.remove(whale_1)
                    if len(valid_indices) > 1:
                        valid_indices.remove(whale_1)
                    whale_2 = valid_indices[random.randint(0, len(valid_indices)-1)]

                    for i in range(int(clicks_per_convo/2)):
                        click_whale_1 = sorted_conversations[whale_1].pop(random.randint(0,len(sorted_conversations[whale_1])-1))
                        click_whale_2 = sorted_conversations[whale_2].pop(random.randint(0,len(sorted_conversations[whale_2])-1))
                        ordered_audio_windows.append(click_whale_1)
                        ordered_audio_windows.append(click_whale_2)

                    noise_example_1 = noise_copy.pop(random.randint(0, len(noise_copy)-1))
                    ordered_audio_windows.append(noise_example_1)
                    noise_example_2 = noise_copy.pop(random.randint(0, len(noise_copy)-1))
                    ordered_audio_windows.append(noise_example_2)

                    hard_noise_example_1 = hard_noise_copy.pop(random.randint(0, len(hard_noise_copy)-1))
                    ordered_audio_windows.append(hard_noise_example_1)
                    hard_noise_example_2 = hard_noise_copy.pop(random.randint(0, len(hard_noise_copy)-1))
                    ordered_audio_windows.append(hard_noise_example_2)
                    hard_noise_example_3 = hard_noise_copy.pop(random.randint(0, len(hard_noise_copy)-1))
                    ordered_audio_windows.append(hard_noise_example_3)
                    hard_noise_example_4 = hard_noise_copy.pop(random.randint(0, len(hard_noise_copy)-1))
                    ordered_audio_windows.append(hard_noise_example_4)
                    hard_noise_example_5 = hard_noise_copy.pop(random.randint(0, len(hard_noise_copy)-1))
                    ordered_audio_windows.append(hard_noise_example_5)
                    hard_noise_example_6 = hard_noise_copy.pop(random.randint(0, len(hard_noise_copy)-1))
                    ordered_audio_windows.append(hard_noise_example_6)

                    new_convo = []
                    for whale_num, clicks in sorted_conversations.items():
                        new_convo += clicks
                    if len(new_convo) >= clicks_per_convo:
                        new_conversations.append(new_convo)
        conversations = new_conversations
                        
    return ordered_audio_windows

   
def annotation_inside(window, annotation_array, whale_numbers):
    def inside(time_step, window):
        return window[0] <= time_step and window[1] > time_step
    window_center = (window[1] - window[0])/2 + window[0]
    click_time = None
    whale_num = None
    for i, time_step in enumerate(annotation_array):
        if inside(time_step, window):
            # click_time = (time_step - window_center)/22.050
            click_time = time_step - window_center
            # click_time = time_step - window[0]
            whale_num = whale_numbers[i]
    return click_time, whale_num

def generate_data(annotation_array, whale_numbers, min_timestep, max_timestep, window_size, black_window_size):
    '''
        annotation_array: array of timesteps indicating positives
        min_timestep: the first timestep in the image
        max_timestep: the final timestep in the image
        window_size: size of the "black" window
        return:
            a collection of data points generated following the natural distribution
    '''
    whale_numbers = [x for _, x in sorted(zip(annotation_array, whale_numbers), key=lambda pair: pair[0])]
    annotation_array.sort()
    window_st = random.randint(min_timestep, max_timestep)
    curr_green_window = [window_st, window_st + window_size]
    black_window_st = window_st + (window_size - black_window_size)/2
    black_window_en = black_window_st + black_window_size

    click_time, whale_num = annotation_inside((black_window_st, black_window_en), annotation_array, whale_numbers)
    return (curr_green_window, click_time, whale_num)

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
        click_times = self.data_in[idx][1]
        whale_numbers = self.data_in[idx][2]

        if audio.shape[1] >= 10040:
            start_index = 5020 - (window_size + black_window_size)/2 + 1
        else:
            start_index = 0
        end_index = start_index + black_window_size - 1
        curr_green_window, click_time, whale_num = generate_data(click_times, whale_numbers, start_index, end_index, window_size, black_window_size)

        if click_time is not None:
            confidence = 1
        else:
            confidence = 0
            click_time = 0
            whale_num = 0
       
        return(audio[:,curr_green_window[0]:curr_green_window[1]], confidence, click_time, whale_num)


def plot_losses_and_accuracy(losses, accuracy, save_folder):
    if not os.path.exists(os.path.join(save_folder, args.exp)):
        os.mkdir(os.path.join(save_folder, args.exp))
    # Training and validation loss
    fig, axs = plt.subplots(2, 1, figsize=(10,20))
    axs[0].plot([i for i in range(len(losses['train_all']))], losses['train_all'], color='k', label='Total Loss')
    axs[0].plot([i for i in range(len(losses['train_confidence']))], losses['train_confidence'], color='r', label='Confidence Loss')
    axs[0].plot([i for i in range(len(losses['train_click_time']))], losses['train_click_time'], color='g', label='Click Time Loss')
    axs[0].plot([i for i in range(len(losses['train_whale_embedding']))], losses['train_whale_embedding'], color='b', label='Whale Embedding Loss')
    axs[0].legend()
    axs[0].set_title('Training Loss')
    axs[0].set_ylabel('Loss')
    axs[1].plot([i for i in range(len(losses['val_all']))], losses['val_all'], color='k', label='Total Loss')
    axs[1].plot([i for i in range(len(losses['val_confidence']))], losses['val_confidence'], color='r', label='Confidence Loss')
    axs[1].plot([i for i in range(len(losses['val_click_time']))], losses['val_click_time'], color='g', label='Click Time Loss')
    axs[1].plot([i for i in range(len(losses['val_whale_embedding']))], losses['val_whale_embedding'], color='b', label='Whale Embedding Loss')
    axs[1].legend()
    axs[1].set_title('Validation Loss')
    axs[1].set_xlabel('Training Epochs')
    axs[1].set_ylabel('Loss')
    plt.savefig(os.path.join(save_folder, args.exp, 'loss.png'))
    plt.close()

    # Validation accuracy
    fig, axs = plt.subplots(figsize=(10,10))
    axs.plot([i for i in range(len(accuracy['val_confidences']))], accuracy['val_confidences'])
    axs.set_title('Validation Confidences Accuracy')
    axs.set_ylabel('Accuracy')
    axs.set_xlabel('Training Epochs')
    plt.savefig(os.path.join(save_folder, args.exp, 'accuracy_confidences.png'))
    plt.close()

    fig, axs = plt.subplots(figsize=(10,10))
    axs.plot([i for i in range(len(accuracy['val_click_times']))], accuracy['val_click_times'])
    axs.set_title('Validation Click Times Error')
    axs.set_ylabel('Error')
    axs.set_xlabel('Training Epochs')
    plt.savefig(os.path.join(save_folder, args.exp, 'error_click_times.png'))
    plt.close()

    

########## Training Code ######################
if __name__ == '__main__':

    # !!!!!!! Important things to set
    number_per_conversation = args.num_per_convo
    whale_embedding_size = args.embedding_size


    np.random.seed(0)
    torch.manual_seed(0)

    # probability_checkpoint = torch.load(os.path.join('probability_detector', 'ckpts', '08_08_22_hard_noise', 'checkpoint.pth.tar'))
    probability_checkpoint = torch.load("/raid/lingo/martinrm/best_models/9_6_2023_full_file_val.pth.tar")
    

    probability_soundnet = SoundNet()
    probability_soundnet.cuda()
    probability_soundnet = nn.DataParallel(probability_soundnet)
    probability_soundnet.load_state_dict(probability_checkpoint['state_dict'])
    probability_soundnet.eval()

    probability_model = probability_net()
    probability_model.cuda()
    probability_model = nn.DataParallel(probability_model)
    probability_model.load_state_dict(probability_checkpoint['state_dict_valnet'])
    probability_model.eval()
    optimizer_probability = optim.Adam(probability_model.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    mlp_model = mlp()  
    mlp_model.cuda()
    mlp_model = nn.DataParallel(mlp_model)
    optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    scheduler_mlp = StepLR(optimizer_mlp, step_size=50, gamma=0.5)

    soundnet_model = SoundNet2()
    soundnet_model.cuda()
    soundnet_model = nn.DataParallel(soundnet_model)
    optimizer_soundnet = optim.Adam(soundnet_model.parameters(), lr=args.lr, weight_decay=args.weightdecay) 
    scheduler_soundnet = StepLR(optimizer_soundnet, step_size=50, gamma=0.5)

    # confidence_model = confidence_net()
    # confidence_model.cuda()
    # confidence_model = nn.DataParallel(confidence_model)
    # optimizer_confidence = optim.Adam(confidence_model.parameters(), lr=args.lr, weight_decay=args.weightdecay) 
    # scheduler_confidence = StepLR(optimizer_confidence, step_size=50, gamma=0.5)

    #click_time_model = click_time_net()
    #click_time_model.cuda()
    #click_time_model = nn.DataParallel(click_time_model)
    #optimizer_click_time = optim.Adam(click_time_model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    #scheduler_click_time = StepLR(optimizer_click_time, step_size=50, gamma=0.5)

    whale_embedding_model = whale_embedding_net1(size=whale_embedding_size)
    whale_embedding_model.cuda()
    whale_embedding_model = nn.DataParallel(whale_embedding_model)
    optimizer_whale_embedding = optim.Adam(whale_embedding_model.parameters(), lr=args.lr, weight_decay=args.weightdecay)  
    scheduler_whale_embedding = StepLR(optimizer_whale_embedding, step_size=50, gamma=0.5)

    class ContrastiveLoss(nn.Module):
        def __init__(self, margin=1.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

        def forward(self, anchor, positive, label):
            distance = nn.PairwiseDistance(p=2)  # Euclidean distance

            euclidean_distance = distance(anchor, positive)
            loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                        (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

            return loss_contrastive
        
    #criterion_probability = nn.CrossEntropyLoss()
    criterion_confidence = nn.CrossEntropyLoss()
    criterion_click_time = nn.MSELoss()
    criterion_whale_embedding = ContrastiveLoss() #edit
    
    # train_dataset = sample_data(audio_recordings_train)
    # val_dataset = sample_data(audio_recordings_val)
    # test_dataset = sample_data(audio_recordings_test)

    # print('Dataset sizes:')
    # print(len(train_dataset), len(val_dataset), len(test_dataset))
        
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)
    # val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True, num_workers=20)
    # test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=20)    
    
    class GaussianDataset(Dataset):
        def __init__(self, num_samples, mean1, std1, mean2, std2):
            self.num_samples = num_samples
            self.mean1 = mean1
            self.std1 = std1
            self.mean2 = mean2
            self.std2 = std2
            self.data, self.labels = self.generate_data()

        def generate_data(self):
            data = []
            labels = []

            for _ in range(self.num_samples):
                if np.random.rand() < 0.5:
                    sample1 = np.random.normal(self.mean1, self.std1, size=(512,))  # Generate a sample with 512 dimensions
                    label = 0  # Label 0 for the first Gaussian dataset
                else:
                    sample2 = np.random.normal(self.mean2, self.std2, size=(512,))  # Generate a sample with 512 dimensions
                    label = 1  # Label 1 for the second Gaussian dataset

                data.append(sample1 if label == 0 else sample2)
                labels.append(label)

            data = torch.Tensor(data)
            labels = torch.LongTensor(labels)

            return data, labels

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            data = self.data[idx]
            labels = self.labels[idx]
            return data, labels

    num_samples = 1000
    mean1, std1 = 0, 1
    mean2, std2 = 8, 2
    
    train_dataset = GaussianDataset(num_samples, mean1, std1, mean2, std2)
    val_dataset = GaussianDataset(num_samples, mean1, std1, mean2, std2)
    test_dataset = GaussianDataset(num_samples, mean1, std1, mean2, std2)

    # Set up data loaders for both training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    losses = {'train_all' : [],
              'train_confidence' : [],
              'train_click_time' : [],
              'train_whale_embedding' : [],
              'val_all' : [],
              'val_confidence' : [],
              'val_click_time' : [],
              'val_whale_embedding' : []}

    accuracy = {'val_confidences' : [],
                'val_click_times' : []}
    
    chkptno = 0
    min_val_loss = None
        
    ########################  TRAINING LOOP ############################
    
    for epoch in range(args.epoch):
        print('\nSTEP: ', epoch)
        # print(f'Previos Losses: {losses["train"]}')
        probability_soundnet.train()
        probability_model.train()
        soundnet_model.train()
        # confidence_model.train()
        # click_time_model.train()
        whale_embedding_model.train()

        avg_train_loss_all = []
        avg_train_loss_confidence = []
        avg_train_loss_click_time = []
        avg_train_loss_whale_embedding = []

        # train_ordered_audio_windows = order_audio_windows(train_conversations, train_noise, hard_noise_train, number_per_conversation)
        # train_dataset = sample_data(train_ordered_audio_windows)
        # train_dataloader = DataLoader(train_dataset, batch_size=512-(512%(number_per_conversation+8)), shuffle=False, num_workers=20)

        #print('Train dataset size: {}'.format(len(train_dataset)))

        for batch_data, batch_labels in train_dataloader:
            # Create anchor-positive pairs
            anchor = batch_data[0::2]
            positive = batch_data[1::2]
            anchor_labels = batch_labels[0::2]
            positive_labels = batch_labels[1::2]
            anchor = anchor.to('cuda:0')
            positive = positive.to('cuda:0')
            anchor_labels = anchor_labels.to('cuda:0')
            positive_labels = positive_labels.to('cuda:0')

            optimizer_soundnet.zero_grad()
            optimizer_probability.zero_grad()
            optimizer_mlp.zero_grad()
            # optimizer_confidence.zero_grad()
            # optimizer_click_time.zero_grad()
            optimizer_whale_embedding.zero_grad()

            # audio = sample_batched[0].type(torch.cuda.FloatTensor) 
            # confidence = sample_batched[1].type(torch.cuda.LongTensor)
            # click_time = sample_batched[2].type(torch.cuda.FloatTensor)
            
            #whale_number = labels

            #print(whale_number)

            # print(confidence)
            # print(click_time)
            # print(whale_number)
            # print('\n')

            # soundnet_out_small = soundnet_model(audio)
            # soundnet_out = soundnet_out_small.unsqueeze(1)
            # featurizer_1_output_small = probability_model(probability_soundnet(audio))
            # featurizer_1_output = featurizer_1_output_small.unsqueeze(2)

            #print('Probability output shape: {}'.format(featurizer_1_output.shape))
            #print('Modified audio shape: {}'.format(soundnet_out.shape))
            # multiplied = torch.matmul(featurizer_1_output, soundnet_out)
            # #print('Multiplied shape: {}'.format(multiplied.shape))
            # combined_input = torch.flatten(multiplied, start_dim=1)
            #print('Flattened shape: {}'.format(combined_input.shape))
            #raise ValueError('DID IT WORK?!')

            # mlp_out = mlp_model(combined_input)
            # confidence_out = confidence  #confidence_model(mlp_out)
            # click_time_out = click_time #click_time_model(mlp_out)
        
            # inputs = inputs.view(-1, 512)
            # whale_embedding_out = whale_embedding_model(inputs)

            #confidence_loss = criterion_confidence(confidence_out, confidence)
            confidence_loss = 0 #confidence_loss*2

            #mask = whale_number.ge(1)
            # click_time_out = click_time_out.flatten()
            # click_time_out_masked = torch.masked_select(click_time_out, mask)
            # click_time_masked = torch.masked_select(click_time, mask)
            #click_time_loss = criterion_click_time(click_time_out_masked, click_time_masked)
            #click_time_loss = click_time_loss/120
            click_time_loss = 0 # click_time_loss*4

            # print(f'True click times: {click_time_masked}')
            # print(f'Predicted click times: {click_time_out_masked}')

            anchor_embeddings = whale_embedding_model(anchor)
            positive_embeddings = whale_embedding_model(positive)
            whale_embedding_loss = criterion_whale_embedding(anchor_embeddings, positive_embeddings, (anchor_labels == positive_labels).float())

            # out_lists = []
            # label_lists = []
            # cuda_device = torch.device('cuda:0')
            # to_compare = whale_embedding_out[0:number_per_conversation].unsqueeze(1)
            # whale_embedding_losses = [criterion_whale_embedding(to_compare, whale_number[:number_per_conversation])/(args.num_per_convo/4)]
            # for i in range(number_per_conversation, whale_embedding_out.shape[0], number_per_conversation):
            #     if i < whale_embedding_out.shape[0] - number_per_conversation:
            #         to_compare = whale_embedding_out[i:i+number_per_conversation].unsqueeze(1)
            #         to_add = criterion_whale_embedding(to_compare, whale_number[i:i+number_per_conversation])/(args.num_per_convo/4)
            #         # # print(f'Conversations embedding loss: {to_add.data.item()}')
            #         # if math.isnan(to_add.data.item()) or math.isinf(to_add.data.item()):
            #         #     print(to_add.data.item())
            #         #     print(to_compare)
            #         #     print(whale_number[i:i+number_per_conversation])
            #         #     raise ValueError('Why is it nan or inf!!!!!')
            #         # whale_embedding_losses.append(to_add)
            # whale_embedding_loss = sum([individual_loss for individual_loss in whale_embedding_losses])
            
            
            # percentile_list = pd.DataFrame(
            #     {'out': out_lists,
            #     'label': label_lists,
            #     })
            
            # percentile_list.to_csv('/raid/lingo/leejason/automatic-annotation/src/temp/whale_label_out_compare.csv')

            if type(whale_embedding_loss) is int:
                print(f'Whale embedding loss: {whale_embedding_loss}')
                print(len(whale_embedding_losses))
                raise ValueError('WHY!!!!!!!!')


            # print(f'Confidence loss: {confidence_loss.data.item()}, Click time loss: {click_time_loss.data.item()}, SupConLoss: {whale_embedding_loss.data.item()}')
            #loss = (whale_embedding_loss)
            # print(f'Overall loss: {loss.data.item()}')
            # loss.backward(retain_graph=True)
            # optimizer_soundnet.step()
            # optimizer_mlp.step()

            # optimizer_click_time.zero_grad()
            # click_time_loss.backward(retain_graph=True)
            # optimizer_click_time.step()

            # optimizer_confidence.zero_grad()
            # confidence_loss.backward(retain_graph=True)
            # optimizer_confidence.step()

            whale_embedding_loss.backward(retain_graph=True)
            optimizer_whale_embedding.step()

            optimizer_soundnet.step()
            optimizer_mlp.step()

            avg_train_loss_all.append(whale_embedding_loss.data.item())
            avg_train_loss_confidence.append(confidence_loss) #confidence_loss.data.item())
            avg_train_loss_click_time.append(click_time_loss) #click_time_loss.data.item())
            avg_train_loss_whale_embedding.append(whale_embedding_loss.data.item())

            # print(f'Batch number: {i_batch}, Confidence loss: {confidence_loss.data.item()}, Click time loss: {click_time_loss.data.item()}, Whale embedding loss: {whale_embedding_loss.data.item()}')
            # if math.isnan(whale_embedding_loss.data.item()) or math.isinf(whale_embedding_loss.data.item()):
            #     print(whale_embedding_loss.data.item())
            #     print(whale_embedding_out)
            #     print(whale_number)
            #     raise ValueError('Why is it nan or inf!!!!!')

        # scheduler_click_time.step()
        # scheduler_confidence.step()
        scheduler_whale_embedding.step()
        # scheduler_soundnet.step()
        # scheduler_mlp.step()

        avg_train_loss = np.mean(avg_train_loss_all)
        train_writer.add_scalar('data/loss', np.mean(avg_train_loss) , chkptno)        
        print('   Average train loss:', avg_train_loss)
        losses['train_all'].append(avg_train_loss)
        losses['train_confidence'].append(np.mean(avg_train_loss_confidence))
        losses['train_click_time'].append(np.mean(avg_train_loss_click_time))
        losses['train_whale_embedding'].append(np.mean(avg_train_loss_whale_embedding))

        wandb.log({"Train Loss": avg_train_loss})
        wandb.log({"Train Confidence Loss": np.mean(avg_train_loss_confidence)})
        wandb.log({"Train Click Time Loss": np.mean(avg_train_loss_click_time)})
        wandb.log({"Train Whale Embedding Loss": np.mean(avg_train_loss_whale_embedding)})

        print(f'Whale embedding loss: {np.mean(avg_train_loss_whale_embedding)}')

        
        ########################  VALIDATION LOOP ############################
        
        print("\nNow running on val set")
        probability_soundnet.eval()
        probability_model.eval()
        soundnet_model.eval()
        # confidence_model.eval()
        # click_time_model.eval()
        whale_embedding_model.eval()
        
        avg_val_loss_all = []
        avg_val_loss_confidence = []
        avg_val_loss_click_time = []
        avg_val_loss_whale_embedding = []

        # val_ordered_audio_windows = order_audio_windows(val_conversations, val_noise, hard_noise_val, number_per_conversation)
        # val_dataset = sample_data(val_ordered_audio_windows)
        # val_dataloader = DataLoader(val_dataset, batch_size=512-(512%(number_per_conversation+8)), shuffle=False, num_workers=20)

        print(f'Val dataset size: {len(val_dataset)}')

        acc_list = []
        error_list = []
        bad_click_times = []

        for batch_data, batch_labels in val_dataloader:
            # audio = sample_batched[0].type(torch.cuda.FloatTensor) 
            # confidence = sample_batched[1].type(torch.cuda.LongTensor)
            # click_time = sample_batched[2].type(torch.cuda.FloatTensor)
            #whale_number = labels

            # soundnet_out_small = soundnet_model(audio)
            # soundnet_out = soundnet_out_small.unsqueeze(1)
            # featurizer_1_output_small = probability_model(probability_soundnet(audio))
            # featurizer_1_output = featurizer_1_output_small.unsqueeze(2)

            # combined_input = torch.flatten(torch.matmul(featurizer_1_output, soundnet_out), start_dim=1)

            # mlp_out = mlp_model(combined_input)
            # confidence_out = confidence #confidence_model(mlp_out)
            # click_time_out = click_time #click_time_model(mlp_out)
            # inputs = inputs.view(-1, 512)
            # whale_embedding_out = whale_embedding_model(inputs)

            anchor = batch_data[0::2]
            positive = batch_data[1::2]
            anchor_labels = batch_labels[0::2]
            positive_labels = batch_labels[1::2]
            anchor = anchor.to('cuda:0')
            positive = positive.to('cuda:0')
            anchor_labels = anchor_labels.to('cuda:0')
            positive_labels = positive_labels.to('cuda:0')
            
            # confidence_loss = criterion_confidence(confidence_out, confidence)
            confidence_loss = 0 # confidence_loss*2

            # mask = whale_number.ge(1)
            # click_time_out_small = click_time_out.flatten()
            # click_time_out_masked = torch.masked_select(click_time_out_small, mask)
            # click_time_masked = torch.masked_select(click_time, mask)
            # click_time_loss = criterion_click_time(click_time_out_masked, click_time_masked)
            # click_time_loss = click_time_loss/120
            click_time_loss = 0 # click_time_loss*4

            # masked_confidence = torch.masked_select(confidence, mask)

            # to_compare = whale_embedding_out[0:number_per_conversation].unsqueeze(1)
            # whale_embedding_losses = [criterion_whale_embedding(to_compare, whale_number[:number_per_conversation])/(args.num_per_convo/4)]

            # for i in range(number_per_conversation, whale_embedding_out.shape[0], number_per_conversation):
            #     if i < whale_embedding_out.shape[0] - number_per_conversation:
            #         to_compare = whale_embedding_out[i:i+number_per_conversation].unsqueeze(1)
            #         to_add = criterion_whale_embedding(to_compare, whale_number[i:i+number_per_conversation])/(args.num_per_convo/4)
            #         # print(f'Conversations embedding loss: {to_add.data.item()}')
            #         if math.isnan(to_add.data.item()) or math.isinf(to_add.data.item()):
            #             print(to_add.data.item())
            #             print(to_compare)
            #             print(whale_number[i:i+number_per_conversation])
            #             raise ValueError('Why is it nan or inf!!!!!')
            #         whale_embedding_losses.append(to_add)
            # whale_embedding_loss = sum([individual_loss for individual_loss in whale_embedding_losses])

            anchor_embeddings = whale_embedding_model(anchor)
            positive_embeddings = whale_embedding_model(positive)
            whale_embedding_loss = criterion_whale_embedding(anchor_embeddings, positive_embeddings, (anchor_labels == positive_labels).float())
            # if math.isnan(whale_embedding_loss.data.item()) or math.isinf(whale_embedding_loss.data.item()):
            #     print(whale_embedding_loss.data.item())
            #     print(whale_embedding_out)
            #     print(whale_number)
            #     raise ValueError('Why is it nan or inf!!!!!')

            avg_val_loss_all.append(whale_embedding_loss.data.item())
            avg_val_loss_confidence.append(confidence_loss) #confidence_loss.data.item())
            avg_val_loss_click_time.append(click_time_loss) #.data.item())
            avg_val_loss_whale_embedding.append(whale_embedding_loss.data.item())

            # confidence_out = confidence_out.cpu().data.numpy()
            # confidence_out = list(confidence.cpu().data.numpy()) # np.argmax(confidence_out,axis=1)
            # #print(confidence)
            # confidence = list(confidence.cpu().data.numpy())
            # acc_temp = 0
            # for i in range(len(confidence_out)): #confidence_out.shape[0]):
            #     if confidence[i] == confidence_out[i]:
            #         acc_temp += 1
            # acc_list.append(acc_temp/len(confidence_out)) #confidence_out.shape[0])

            # click_time_out = click_time_out_masked.cpu().data.numpy()
            # click_time = click_time_masked.cpu().data.numpy()
            # masked_confidence = masked_confidence.cpu().data.numpy()
            # error_temp = 0
            # total_clicks = 0
            # # num_printed = 0
            # for i in range(click_time_out.shape[0]):
            #     if masked_confidence[i] == 1:
            #         # if i_batch == 0 and num_printed < 100:
            #         #     print(f'Predicted click time: {click_time_out[i][0]}, Actual click time: {click_time[i]}')
            #         #     num_printed += 1
            #         error_temp += abs(click_time_out[i] - click_time[i])
            #         total_clicks += 1
            #         if abs(click_time_out[i] - click_time[i]) > 2:
            #             bad_click_times.append((click_time_out[i], click_time[i]))
            # error_list.append(error_temp/total_clicks)

        val_loss = np.mean(avg_val_loss_all)
        print('   Average val loss:', val_loss)
        print(f'  Whale embedding loss: {np.mean(avg_val_loss_whale_embedding)}')

        val_writer.add_scalar('data/loss', val_loss, chkptno)
        losses['val_all'].append(val_loss)
        losses['val_confidence'].append(np.mean(avg_val_loss_confidence))
        losses['val_click_time'].append(np.mean(avg_val_loss_click_time))
        losses['val_whale_embedding'].append(np.mean(avg_val_loss_whale_embedding))

        wandb.log({'val_loss': val_loss, 
                   'val_confidence_loss': np.mean(avg_val_loss_confidence), 
                   'val_click_time_loss': np.mean(avg_val_loss_click_time),
                   'val_whale_embedding_loss': np.mean(avg_val_loss_whale_embedding)})

        # print(f'Number of bad clicks: {len(bad_click_times)}')
        # random.shuffle(bad_click_times)
        # for i in range(min(20, len(bad_click_times))):
        #     print(f'    Predicted time: {bad_click_times[i][0]}, Actual time: {bad_click_times[i][1]}')

        if chkptno == 0:
            min_val_loss = val_loss

        if val_loss <= min_val_loss:
            min_val_loss = val_loss

            save_checkpoint({
                'epoch': chkptno,
                'mlp_model': mlp_model.state_dict(),
                'soundnet_model' : soundnet_model.state_dict(),
                #'confidence_model' : confidence_model.state_dict(),
                #'click_time_model' : click_time_model.state_dict(),
                'whale_embedding_model' : whale_embedding_model.state_dict(),
            }, chkptno)

        acc = np.mean(acc_list)
        error = np.mean(error_list)

        accuracy['val_confidences'].append(acc)
        print("   Accuracy val confidences:", acc)

        accuracy['val_click_times'].append(error)
        print("   Error val click times:", error)

        # val_writer.add_scalar('data/acc', acc/labels_out.shape[0], chkptno)

        chkptno = chkptno+1
        
    ########################  TESTING LOOP ############################

    # Load in best checkpoint model
    checkpoint = torch.load(os.path.join(save_folder, 'ckpts', args.exp, 'checkpoint.pth.tar'))

    mlp_model = mlp()
    mlp_model.cuda()
    mlp_model = nn.DataParallel(mlp_model)
    mlp_model.load_state_dict(checkpoint['mlp_model'])
    mlp_model.eval()

    # probability_model = probability_net()
    # probability_model.cuda()
    # probability_model = nn.DataParallel(probability_model)
    # probability_model.load_state_dict(checkpoint['probability_model'])
    # probability_model.eval()

    soundnet_model = SoundNet2()
    soundnet_model.cuda()
    soundnet_model = nn.DataParallel(soundnet_model)
    soundnet_model.load_state_dict(checkpoint['soundnet_model'])
    soundnet_model.eval()

    # confidence_model = confidence_net()
    # confidence_model.cuda()
    # confidence_model = nn.DataParallel(confidence_model)
    # confidence_model.load_state_dict(checkpoint['confidence_model'])
    # confidence_model.eval()

    # click_time_model = click_time_net()
    # click_time_model.cuda()
    # click_time_model = nn.DataParallel(click_time_model)
    # click_time_model.load_state_dict(checkpoint['click_time_model'])
    # click_time_model.eval()

    whale_embedding_model = whale_embedding_net1(size=whale_embedding_size)
    whale_embedding_model.cuda()
    whale_embedding_model = nn.DataParallel(whale_embedding_model)
    whale_embedding_model.load_state_dict(checkpoint['whale_embedding_model'])
    whale_embedding_model.eval()

    # test_ordered_audio_windows = order_audio_windows(test_conversations, test_noise, hard_noise_test, number_per_conversation)
    # test_dataset = sample_data(test_ordered_audio_windows)
    # test_dataloader = DataLoader(test_dataset, batch_size=512-(512%(number_per_conversation+8)), shuffle=False, num_workers=20) 

    print(f'Test dataset size: {len(test_dataset)}')

    acc_list = []
    error_list = []
    
    avg_test_loss_whale_embedding = []
    correct_predictions = 0
    total_samples = 0
    total_predictions = 0

    class BinaryClassificationHead(nn.Module):
        def __init__(self, input_size):
            super(BinaryClassificationHead, self).__init__()
            self.fc = nn.Linear(input_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc(x)
            x = self.sigmoid(x)
            return x
    
    count = 0
    for inputs, labels in test_dataloader:
        # audio = sample_batched[0].type(torch.cuda.FloatTensor) 
        # confidence = sample_batched[1].type(torch.cuda.LongTensor)
        # click_time = sample_batched[2].type(torch.cuda.FloatTensor)
        whale_number = labels #sample_batched[3].type(torch.cuda.LongTensor)

        # soundnet_out_small = soundnet_model(audio)
        # soundnet_out = soundnet_out_small.unsqueeze(1)
        # featurizer_1_output_small = probability_model(probability_soundnet(audio))
        # featurizer_1_output = featurizer_1_output_small.unsqueeze(2)

        # combined_input = torch.flatten(torch.matmul(featurizer_1_output, soundnet_out), start_dim=1)

        # mlp_out = mlp_model(combined_input)
        # confidence_out = confidence #confidence_model(mlp_out)
        # click_time_out = click_time #click_time_model(mlp_out)
        inputs = inputs.view(-1, 512)
        whale_embedding_out = whale_embedding_model(inputs)

        # mask = whale_number.ge(1)
        # click_time_out_small = click_time_out.flatten()
        # click_time_out_masked = torch.masked_select(click_time_out_small, mask)
        # click_time_masked = torch.masked_select(click_time, mask)
        # masked_confidence = torch.masked_select(confidence, mask)

        # confidence_out = confidence_out.cpu().data.numpy()
        # confidence_out = list(confidence.cpu().data.numpy()) #np.argmax(confidence_out,axis=1)
        # confidence = list(confidence.cpu().data.numpy())
        # acc_temp = 0
        # for i in range(len(confidence_out)): #confidence_out.shape[0]):
        #     if confidence[i] == confidence_out[i]:
        #         acc_temp += 1
        # acc_list.append(acc_temp/len(confidence_out)) #confidence_out.shape[0])

        # click_time_out = click_time_out_masked.cpu().data.numpy()
        # click_time = click_time_masked.cpu().data.numpy()
        # masked_confidence = masked_confidence.cpu().data.numpy()
        # error_temp = 0
        # total_clicks = 0
        # for i in range(click_time_out.shape[0]):
        #     if masked_confidence[i] == 1:
        #         error_temp += abs(click_time_out[i] - click_time[i])
        #         total_clicks += 1
        # error_list.append(error_temp/total_clicks)

        # to_compare = whale_embedding_out[0:number_per_conversation].unsqueeze(1)
        # whale_embedding_losses = [criterion_whale_embedding(to_compare, whale_number[:number_per_conversation]) / (args.num_per_convo / 4)]
        # for i in range(number_per_conversation, whale_embedding_out.shape[0], number_per_conversation):
        #     if i < whale_embedding_out.shape[0] - number_per_conversation:
        #         to_compare = whale_embedding_out[i:i + number_per_conversation].unsqueeze(1)
        #         to_add = criterion_whale_embedding(to_compare, whale_number[i:i + number_per_conversation]) / (args.num_per_convo / 4)
        #         whale_embedding_losses.append(to_add)

        # whale_embedding_loss = sum(whale_embedding_losses)

        # avg_test_loss_whale_embedding.append(whale_embedding_loss.data.item())

        # Calculate predictions (assuming binary classification)
        binary_head = BinaryClassificationHead(whale_embedding_size)

        labels = labels.to('cuda:0')
        binary_head = binary_head.to('cuda:0')

        # Apply the head to your embeddings
        
        predictions = binary_head(whale_embedding_out)
        threshold = 0.5 # Adjust this threshold as needed
        binary_predictions = (predictions >= threshold).to(torch.int)

        labels = labels.view(-1)  # Flatten the labels if necessary

        # Update accuracy metrics
        total_predictions += len(binary_predictions)
    
        binary_predictions = binary_predictions.squeeze()

        correct_predictions += (binary_predictions == labels).sum().item()
        total_samples += len(labels)

        # Perform PCA on both labels and predictions
        n_components = 2 # You can choose the number of components (2 for a 2D plot)
        pca = PCA(n_components=n_components)

        #labels_pca = pca.fit_transform(labels.cpu().numpy().reshape(-1, 1))
        predictions_pca = pca.fit_transform(whale_embedding_out.cpu().detach().numpy())

        # Calculate whether predictions are correct or not
        correct_or_incorrect = ((binary_predictions == labels).cpu().numpy()).astype(int)
        colors = ['red' if is_correct == 0 else 'blue' for is_correct in correct_or_incorrect]

        # Create a scatter plot
        plt.figure(figsize=(8, 6))
        #plt.scatter(labels_pca[:, 0], labels_pca[:, 1], c='blue', label='Labels', alpha=0.5)
        plt.scatter(predictions_pca[:, 0], predictions_pca[:, 1], c=colors, label='Predictions', alpha=0.5)

        plt.title('PCA Comparison of Labels and Predictions')
        plt.xlabel(f'Principal Component 1')
        plt.ylabel(f'Principal Component 2')
        plt.legend()
        plt.grid()

        plt.savefig('/raid/lingo/leejason/automatic-annotation/src/prcurve_testing/pca_visualization' + str(count) + '.png')
        count += 1

        acc = correct_predictions / total_samples
        error = np.mean(avg_test_loss_whale_embedding)

    wandb.log({"Test Accuracy": acc})
    wandb.log({"Test Error": error})


    print("   Accuracy test confidences:", acc)
    print("   Error test click times:", error)

    # param_grid = {
    # 'input_size': [your_input_size],
    # 'hidden_size': [32, 64, 128],
    # 'output_size': [your_output_size],
    # 'learning_rate': [0.001, 0.01, 0.1],
    # 'batch_size': [16, 32, 64],
    # 'epochs': [10, 20, 30]
    # }

    # # Create a GridSearchCV object
    # grid_search = GridSearchCV(estimator=None, param_grid=param_grid, scoring='accuracy', cv=3)

    # # Start the hyperparameter search
    # grid_search.fit(train_data, train_labels)

    # # Print the best hyperparameters and the corresponding accuracy
    # print("Best Hyperparameters: ", grid_search.best_params_)
    # print("Best Accuracy: ", grid_search.best_score_)

    # cuda_device = torch.device('cuda:0')
    plot_losses_and_accuracy(losses, accuracy, plot_save_folder)
    # save_path = os.path.join(save_folder, args.exp, 'precision_recall.png')

    # # Assuming whale_number is a tensor with multi-class labels
    # whale_number = labels
    # whale_embedding_out = outs
    # print(len(whale_number))
    # print(len(whale_embedding_out))

    # # Get the unique class labels
    # unique_classes = np.unique(whale_number)

    # # Calculate precision and recall for each class
    # plt.figure()

    # for class_label in unique_classes:
    #     binary_labels = (whale_number == class_label).astype(int)
    #     precision, recall, _ = precision_recall_curve(binary_labels, whale_embedding_out)
    #     plt.step(recall, precision, where='post', label=f'Class {class_label}')

    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall Curve')
    # plt.legend(loc='best')
    # plt.savefig(plot_save_folder)

    #plot_precision_recall_curve(label, out, plot_save_folder)
    
