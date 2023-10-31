from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import argparse
import librosa
import numpy as np
import os
import pickle
import random
import shutil
import torch
import time
import math

from models.soundNet.soundNet import SoundNet, SoundNet2
from models.soundNet.soundNet_outputs import probability_net, confidence_net, click_time_net, whale_embedding_net
from models.soundNet.soundNet_outputs import confidence_net_transformer, click_time_net_transformer, whale_embedding_net_transformer
from models.soundNet.mlp_output import mlp
from models.vision_transformer.vision_transformer import ViT
from eval.plot_precision_recall import plot_precision_recall_curve
from losses.supervised_contrastive_loss import SupConLoss

##################################################
# Training command
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train_transformer.py --exp 08_12_22_smaller_lr --lr 2e-5 --epoch 100 --samp 32

######## Arguments to run the code ###############

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "11"

assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == torch.cuda.device_count(), "Number of cuda devices not equal to number of visible devices"


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp', type=str, required = True, help='name of experiment')
parser.add_argument('--lr', type=float, default=1e-3, help='name of experiment')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--samp', type=int, default=128, help='Sample per update')
parser.add_argument('--weightdecay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epoch', type=int, default=500, help='The time steps you want to subsample the dataset to')
parser.add_argument('--home_folder', type= str, default='/data/vision/torralba/scratch/fjacob/simplified_transformer/transformer/', help='Where are you training your model')
parser.add_argument('--checkpt', type= str, default='23_07_21/1.pth.tar', help='check point')

args = parser.parse_args()

save_folder = 'transformer'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

plot_save_folder = './transformer/plots'
if not os.path.exists(plot_save_folder):
    os.makedirs(plot_save_folder)

if not os.path.exists('./transformer/ckpts'):
    os.makedirs('./transformer/ckpts')

if not os.path.exists(os.path.join('./transformer/ckpts', args.exp)):
    os.makedirs(os.path.join('./transformer/ckpts', args.exp))

if not os.path.exists('./transformer/tbs'):
    os.makedirs('./transformer/tbs')
    
train_writer = SummaryWriter(os.path.join('./transformer/tbs', args.exp , 'train'))
val_writer = SummaryWriter(os.path.join('./transformer/tbs', args.exp, 'val'))

def save_checkpoint(state, epoch, best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join('transformer', 'ckpts', args.exp, '{}.pth.tar'.format(str(epoch))))
    if best:
        shutil.copyfile(os.path.join('transformer', 'ckpts', args.exp, '{}.pth.tar'.format(str(epoch))), os.path.join('transformer', 'ckpts', args.exp, filename))

###### Dataset Loading and Splitting ##########

train_conversations = pickle.load(open('/data/vision/torralba/scratch/fjacob/simplified_transformer/dataset/transformer_train_conversations_short.p',"rb"))
val_conversations = pickle.load(open('/data/vision/torralba/scratch/fjacob/simplified_transformer/dataset/transformer_test_conversations_short.p',"rb"))
test_conversations = pickle.load(open('/data/vision/torralba/scratch/fjacob/simplified_transformer/dataset/transformer_val_conversations_short.p',"rb"))

######################## Helper functions ###################### 


def annotation_inside(window, annotation_array, whale_numbers, click_in_coda):
    def inside(time_step, window):
        return window[0] <= time_step and window[1] > time_step
    window_center = (window[1] - window[0])/2 + window[0]
    click_time = None
    whale_num = None
    click_num = None
    for i, time_step in enumerate(annotation_array):
        if inside(time_step, window):
            click_time = time_step - window_center
            whale_num = whale_numbers[i]
            click_num = click_in_coda[i]
    return click_time, whale_num, click_num


class sample_data(Dataset):
    def __init__(self, data_in): 
        self.data_in = data_in

    def __len__(self):
        return len(self.data_in)
    
    def __getitem__(self, idx):
        window_size = 1000
        black_window_size = 40
        offset = random.randint(0, black_window_size)

        audios = []
        confidences = []
        click_times = []
        whale_nums = []
        click_nums = []
        data_mask = []

        conversations = self.data_in[idx][:3000]
        num_included = 0

        for i in range(len(conversations)):

            # audio,sr = librosa.load(conversations[i][0],mono=False)
            click_times_all = conversations[i][1]
            whale_numbers = conversations[i][2]
            click_in_coda = conversations[i][3]

            # audio_cropped = audio[:, offset+4500:offset+window_size+4500]
            black_window_start = offset + (window_size - black_window_size)/2 + 4500
            click_time, whale_num, click_num = annotation_inside((black_window_start, black_window_start + black_window_size), click_times_all, whale_numbers, click_in_coda)

            include = False
            if click_time is not None:
                confidence = 1
                data_mask.append(True)
                include = True
                num_included += 1
                audio,sr = librosa.load(conversations[i][0],mono=False)
                audio_cropped = audio[:, offset+4500:offset+window_size+4500]
            elif random.random() < 0.1:
                confidence = 0
                click_time = 0
                whale_num = 0
                click_num = 0
                data_mask.append(True)
                include = True
                num_included += 1
                audio,sr = librosa.load(conversations[i][0],mono=False)
                audio_cropped = audio[:, offset+4500:offset+window_size+4500]
            else:
                data_mask.append(False)
                include = False

            if include:
                audios.append(audio_cropped)
                confidences.append(confidence)
                click_times.append(click_time)
                whale_nums.append(whale_num)
                click_nums.append(click_num)
        # print('Num included: {}, len audios: {}'.format(num_included, len(audios)))
       
        return (torch.tensor(audios), torch.tensor(confidences), torch.tensor(click_times), torch.tensor(whale_nums), torch.tensor(data_mask), torch.tensor(click_nums))


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

    whale_embedding_size = 20

    np.random.seed(0)
    torch.manual_seed(0)

    probability_checkpoint = torch.load(os.path.join('/raid/lingo/martinrm/best_models/9_6_2023_full_file_val.pth.tar'))

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

    featurizer_checkpoint =  torch.load("/raid/lingo/martinrm/automatic-annotation/src/featurizer/ckpts/good_prob_model_checkpoint/checkpoint.pth.tar")

    mlp_model = mlp()  
    mlp_model.cuda()
    mlp_model = nn.DataParallel(mlp_model)
    mlp_model.load_state_dict(featurizer_checkpoint['mlp_model'])
    mlp_model.eval()

    soundnet_model = SoundNet2()
    soundnet_model.cuda()
    soundnet_model = nn.DataParallel(soundnet_model)
    soundnet_model.load_state_dict(featurizer_checkpoint['soundnet_model'])
    soundnet_model.eval()

    featurizer_confidence_model = confidence_net()
    featurizer_confidence_model.cuda()
    featurizer_confidence_model = nn.DataParallel(featurizer_confidence_model)
    featurizer_confidence_model.load_state_dict(featurizer_checkpoint['confidence_model'])
    featurizer_confidence_model.eval()

    featurizer_click_time_model = click_time_net()
    featurizer_click_time_model.cuda()
    featurizer_click_time_model = nn.DataParallel(featurizer_click_time_model)
    featurizer_click_time_model.load_state_dict(featurizer_checkpoint['click_time_model'])
    featurizer_click_time_model.eval()

    featurizer_whale_embedding_model = whale_embedding_net(size=whale_embedding_size)
    featurizer_whale_embedding_model.cuda()
    featurizer_whale_embedding_model = nn.DataParallel(featurizer_whale_embedding_model)
    featurizer_whale_embedding_model.load_state_dict(featurizer_checkpoint['whale_embedding_model'])
    featurizer_whale_embedding_model.eval()

    # New models that will actually be trained...
    transformer_model = ViT(unit_size=24, dim=1024, depth=6, heads=8, mlp_dim=2048, dim_head = 64, dropout = 0.1, emb_dropout = 0.1)
    transformer_model.cuda()
    transformer_model = nn.DataParallel(transformer_model)
    optimizer_transformer = optim.Adam(transformer_model.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    confidence_model = confidence_net_transformer()
    confidence_model.cuda()
    confidence_model = nn.DataParallel(confidence_model)
    optimizer_confidence = optim.Adam(confidence_model.parameters(), lr=args.lr, weight_decay=args.weightdecay) 

    click_time_model = click_time_net_transformer()
    click_time_model.cuda()
    click_time_model = nn.DataParallel(click_time_model)
    optimizer_click_time = optim.Adam(click_time_model.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    whale_embedding_model = whale_embedding_net_transformer(size=whale_embedding_size)
    whale_embedding_model.cuda()
    whale_embedding_model = nn.DataParallel(whale_embedding_model)
    optimizer_whale_embedding = optim.Adam(whale_embedding_model.parameters(), lr=args.lr, weight_decay=args.weightdecay)  
    
    criterion_confidence = nn.CrossEntropyLoss()
    criterion_click_time = nn.MSELoss()
    criterion_whale_embedding = SupConLoss()  
    
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

    random.shuffle(train_conversations)
    train_dataset = sample_data(train_conversations)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=16)
    print('Number of training conversations: {}'.format(len(train_dataset)))

    val_dataset = sample_data(val_conversations)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16)
    print('Number of validation conversations: {}'.format(len(val_dataset)))

    test_dataset = sample_data(test_conversations)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)
    print('Number of testing conversations: {}'.format(len(test_dataset)))
        
    ########################  TRAINING LOOP ############################

    samples_per_epoch = args.samp
    max_transformer_input_length = 40
    
    for epoch in range(args.epoch):
        print('\nSTEP: ', epoch)
        transformer_model.train()
        confidence_model.train()
        click_time_model.train()
        whale_embedding_model.train()

        avg_train_loss_all = []
        avg_train_loss_confidence = []
        avg_train_loss_click_time = []
        avg_train_loss_whale_embedding = []
        
        transformer_input_tensor = None

        for i_batch, sample_batched in enumerate(train_dataloader):
            # print(f'Batch: {i_batch}, Total Allocated: {torch.cuda.memory_allocated(0)/(1024*1024)} MiB')

            audio = sample_batched[0][0].type(torch.cuda.FloatTensor) 
            confidence = sample_batched[1][0].type(torch.cuda.LongTensor) 
            click_time = sample_batched[2][0].type(torch.cuda.FloatTensor) 
            whale_number = sample_batched[3][0].type(torch.cuda.LongTensor) 
            data_mask = sample_batched[4][0].type(torch.cuda.BoolTensor)

            # print(f'After loading in data, Total Allocated: {torch.cuda.memory_allocated(0)/(1024*1024)} MiB')
            # start_time = time.time()
            soundnet_out = soundnet_model(audio)
            soundnet_out = soundnet_out.unsqueeze(1)
            # soundnet_time = time.time() - start_time

            # start_time = time.time()
            featurizer_1_output = probability_model(probability_soundnet(audio))
            featurizer_1_output = featurizer_1_output.unsqueeze(2)
            combined_input = torch.flatten(torch.matmul(featurizer_1_output, soundnet_out), start_dim=1)
            del soundnet_out
            del featurizer_1_output
            del audio

            if not combined_input.is_cuda:
                raise ValueError('Combined input is not on GPU...')

            mlp_out = mlp_model(combined_input)
            featurizer_confidence_out = featurizer_confidence_model(mlp_out)
            featurizer_click_time_out = featurizer_click_time_model(mlp_out)
            featurizer_whale_embedding_out = featurizer_whale_embedding_model(mlp_out)
            # featurizer_time = time.time() - start_time
            del combined_input
            del mlp_out

            # conversation_length = min(featurizer_confidence_out.shape[0], 4000)
            # conversation_length = featurizer_confidence_out.shape[0]

            # print(f'After featurizer, Total Allocated: {torch.cuda.memory_allocated(0)/(1024*1024)} MiB')

            # Then need to filter out all of the low confidence windows
            softmax_featurizer_confidence_out = nn.functional.softmax(featurizer_confidence_out, dim=1)[:,1]
            softmax_featurizer_confidence_out = softmax_featurizer_confidence_out.unsqueeze(1)
            low_confidence_mask = softmax_featurizer_confidence_out.ge(0.1).flatten()  # TODO: Figure out if this is a good enough threshold

            masked_confidence = confidence[low_confidence_mask]
            masked_click_time = click_time[low_confidence_mask]
            masked_whale_number = whale_number[low_confidence_mask]
            del confidence
            del click_time
            del whale_number

            masked_featurizer_confidence_out = featurizer_confidence_out[low_confidence_mask]
            masked_softmax_confidence_out = softmax_featurizer_confidence_out[low_confidence_mask]
            del featurizer_confidence_out

            masked_featurizer_click_time_out = featurizer_click_time_out[low_confidence_mask]
            del featurizer_click_time_out

            masked_featurizer_whale_embedding_out = featurizer_whale_embedding_out[low_confidence_mask]
            del featurizer_whale_embedding_out

            # Then need to combine the inputs of the windows we want to keep

            transformer_input = torch.cat((masked_featurizer_confidence_out, masked_softmax_confidence_out, masked_featurizer_click_time_out, masked_featurizer_whale_embedding_out), dim=-1)
            transformer_input = transformer_input[:40]
            original_length = transformer_input.shape[0]
            zeros = torch.zeros((40-transformer_input.shape[0], transformer_input.shape[1])).type(torch.cuda.BoolTensor)
            zeros.cuda()
            transformer_input = torch.cat((transformer_input, zeros))
            transformer_input = transformer_input.unsqueeze(0)

            del masked_featurizer_confidence_out
            del masked_softmax_confidence_out
            del masked_featurizer_click_time_out
            del masked_featurizer_whale_embedding_out

            new_mask = []
            low_confidence_index = 0
            num_true = 0
            for i in range(data_mask.shape[0]):
                if data_mask[i]:
                    if low_confidence_mask[low_confidence_index] and num_true < 40:
                        new_mask.append(True)
                        num_true += 1
                    else:
                        new_mask.append(False)
                    low_confidence_index += 1
                else:
                    new_mask.append(False)

            combined_mask = torch.tensor(new_mask).type(torch.cuda.BoolTensor)
            ones = torch.ones((40-num_true,)).type(torch.cuda.BoolTensor)
            ones.cuda()
            zeros = torch.zeros((3040-(40-num_true)-combined_mask.shape[0],)).type(torch.cuda.BoolTensor)
            zeros.cuda()
            combined_mask.cuda()

            # print('Combined Mask: {}'.format(combined_mask.is_cuda))
            # print('Ones: {}'.format(ones.is_cuda))
            # print('Zeros: {}'.format(zeros.is_cuda))
            # print('Combined mask cuda: {}, mask shape: {}'.format(combined_mask.is_cuda, combined_mask.shape))
            combined_mask = torch.cat((combined_mask, ones, zeros), dim=0)
            del ones
            del zeros
            del new_mask
            combined_mask = combined_mask.unsqueeze(0)

            if transformer_input_tensor is None:
                transformer_input_tensor = transformer_input
                original_lengths = [original_length] 
                combined_mask_tensor = combined_mask
                combined_masked_confidence = masked_confidence[:40]
                combined_masked_click_time = masked_click_time[:40]
                combined_masked_whale_number = masked_whale_number[:40]
            else:
                transformer_input_tensor = torch.cat((transformer_input_tensor, transformer_input), dim=0)
                original_lengths.append(original_length)
                combined_mask_tensor = torch.cat((combined_mask_tensor, combined_mask), dim=0)
                combined_masked_confidence = torch.cat((combined_masked_confidence, masked_confidence[:40]), dim=0)
                combined_masked_click_time = torch.cat((combined_masked_click_time, masked_click_time[:40]), dim=0)
                combined_masked_whale_number = torch.cat((combined_masked_whale_number, masked_whale_number[:40]), dim=0)

            del combined_mask
            del transformer_input
            del masked_confidence
            del masked_click_time
            del masked_whale_number

            if (i_batch+1)%samples_per_epoch==0:

                transformer_out = transformer_model(transformer_input_tensor, combined_mask_tensor)
                del transformer_input_tensor, combined_mask_tensor

                combined_transformer_out = None
                for i, length in enumerate(original_lengths):
                    if combined_transformer_out is not None:
                        combined_transformer_out = torch.cat((combined_transformer_out, transformer_out[i][:length]))
                    else:
                        combined_transformer_out = transformer_out[i][:length]

                confidence_out = confidence_model(combined_transformer_out)
                click_time_out = click_time_model(combined_transformer_out)
                whale_embedding_out = whale_embedding_model(combined_transformer_out)
                del combined_transformer_out

                # print('Soundnet time: {}, Featurizer time: {}, Transformer time: {}'.format(soundnet_time, featurizer_time, transformer_time))
                # print(f'After transformer, Total Allocated: {torch.cuda.memory_allocated(0)/(1024*1024)} MiB')

                # Then compute loss for the transformer
                confidence_loss = None
                if confidence_out.shape[0] != 0:
                    confidence_loss = (criterion_confidence(confidence_out, combined_masked_confidence)*40)/samples_per_epoch
                del confidence_out
                del combined_masked_confidence

                click_time_loss = None
                mask = combined_masked_whale_number.ge(1)
                click_time_out = click_time_out.flatten()
                click_time_out_masked = torch.masked_select(click_time_out, mask)
                click_time_masked = torch.masked_select(combined_masked_click_time, mask)
                if click_time_masked.shape[0] != 0:
                    click_time_loss = (criterion_click_time(click_time_out_masked, click_time_masked)/3)/samples_per_epoch
                del click_time_out
                del click_time_out_masked
                del click_time_masked

                original_lengths = [0] + original_lengths
                whale_embedding_loss = None
                for i in range(1, len(original_lengths)):
                    start, end = sum(original_lengths[:i]), sum(original_lengths[:i+1])
                    actual_whale_number = combined_masked_whale_number[start:end]
                    actual_whale_number = actual_whale_number[mask[start:end]]
                    actual_whale_embedding_out = whale_embedding_out[start:end]
                    actual_whale_embedding_out = actual_whale_embedding_out[mask[start:end]]
                    if actual_whale_embedding_out.shape[0] > 1:
                        if not whale_embedding_loss:
                            whale_embedding_loss = (criterion_whale_embedding(actual_whale_embedding_out.unsqueeze(1), actual_whale_number)/4)/samples_per_epoch
                        else:
                            whale_embedding_loss += (criterion_whale_embedding(actual_whale_embedding_out.unsqueeze(1), actual_whale_number)/4)/samples_per_epoch
                # del actual_whale_number
                del actual_whale_embedding_out
                del whale_embedding_out

                if click_time_loss:
                    optimizer_click_time.zero_grad()
                    click_time_loss.backward(retain_graph=True)
                    optimizer_click_time.step()

                if confidence_loss:
                    optimizer_confidence.zero_grad()
                    confidence_loss.backward(retain_graph=True)
                    optimizer_confidence.step()

                if whale_embedding_loss:
                    optimizer_whale_embedding.zero_grad()
                    whale_embedding_loss.backward(retain_graph=True)
                    optimizer_whale_embedding.step()

                # print(f'Confidence loss: {confidence_loss.data.item()}, Click time loss: {click_time_loss.data.item()}, Embedding loss: {whale_embedding_loss.data.item()}')

                if math.isinf(confidence_loss.data.item()) or math.isnan(confidence_loss.data.item()) or math.isinf(click_time_loss.data.item()) or math.isnan(click_time_loss.data.item()) or math.isinf(whale_embedding_loss.data.item()) or math.isnan(whale_embedding_loss.data.item()):
                    raise ValueError('Learning rate too big...')

                optimizer_transformer.step()

                optimizer_transformer.zero_grad()
                optimizer_confidence.zero_grad()
                optimizer_click_time.zero_grad()
                optimizer_whale_embedding.zero_grad()

                if click_time_loss:
                    avg_train_loss = confidence_loss.data.item() + click_time_loss.data.item() + whale_embedding_loss.data.item()
                    avg_train_loss_all.append(avg_train_loss)
                    avg_train_loss_click_time.append(click_time_loss.data.item())
                    avg_train_loss_whale_embedding.append(whale_embedding_loss.data.item())
                    avg_train_loss_confidence.append(confidence_loss.data.item())

                    del avg_train_loss
                    del confidence_loss
                    del click_time_loss
                    del whale_embedding_loss

                transformer_input_tensor = None

        avg_train_loss = np.mean(avg_train_loss_all)
        train_writer.add_scalar('data/loss', np.mean(avg_train_loss) , chkptno)        
        print('   Average train loss:', avg_train_loss)
        losses['train_all'].append(avg_train_loss)
        losses['train_confidence'].append(np.mean(avg_train_loss_confidence))
        losses['train_click_time'].append(np.mean(avg_train_loss_click_time))
        losses['train_whale_embedding'].append(np.mean(avg_train_loss_whale_embedding))

        print(f'AVERAGE!! Confidence loss: {np.mean(avg_train_loss_confidence)}, Click time loss: {np.mean(avg_train_loss_click_time)}, Embedding loss: {np.mean(avg_train_loss_whale_embedding)}')
        
        ########################  VALIDATION LOOP ############################
        
        print("\nNow running on val set")
        transformer_model.eval()
        confidence_model.eval()
        click_time_model.eval()
        whale_embedding_model.eval()
        
        avg_val_loss_all = []
        avg_val_loss_confidence = []
        avg_val_loss_click_time = []
        avg_val_loss_whale_embedding = []

        acc_list = []
        error_list = []

        transformer_input_tensor = None

        for i_batch, sample_batched in enumerate(val_dataloader):
            
            audio = sample_batched[0][0].type(torch.cuda.FloatTensor) 
            confidence = sample_batched[1][0].type(torch.cuda.LongTensor) 
            click_time = sample_batched[2][0].type(torch.cuda.FloatTensor) 
            whale_number = sample_batched[3][0].type(torch.cuda.LongTensor) 
            data_mask = sample_batched[4][0].type(torch.cuda.BoolTensor)

            # print(f'After loading in data, Total Allocated: {torch.cuda.memory_allocated(0)/(1024*1024)} MiB')

            soundnet_out = soundnet_model(audio)
            soundnet_out = soundnet_out.unsqueeze(1)
            featurizer_1_output = probability_model(probability_soundnet(audio))
            featurizer_1_output = featurizer_1_output.unsqueeze(2)
            combined_input = torch.flatten(torch.matmul(featurizer_1_output, soundnet_out), start_dim=1)
            del soundnet_out
            del featurizer_1_output
            del audio

            mlp_out = mlp_model(combined_input)
            featurizer_confidence_out = featurizer_confidence_model(mlp_out)
            featurizer_click_time_out = featurizer_click_time_model(mlp_out)
            featurizer_whale_embedding_out = featurizer_whale_embedding_model(mlp_out)
            del combined_input
            del mlp_out

            # conversation_length = min(featurizer_confidence_out.shape[0], 4000)

            # print(f'After featurizer, Total Allocated: {torch.cuda.memory_allocated(0)/(1024*1024)} MiB')

            # Then need to filter out all of the low confidence windows
            softmax_featurizer_confidence_out = nn.functional.softmax(featurizer_confidence_out, dim=1)[:,1]
            softmax_featurizer_confidence_out = softmax_featurizer_confidence_out.unsqueeze(1)
            low_confidence_mask = softmax_featurizer_confidence_out.ge(0.1).flatten()  # TODO: Figure out if this is a good enough threshold

            masked_confidence = confidence[low_confidence_mask]
            masked_click_time = click_time[low_confidence_mask]
            masked_whale_number = whale_number[low_confidence_mask]
            del confidence
            del click_time
            del whale_number

            masked_featurizer_confidence_out = featurizer_confidence_out[low_confidence_mask]
            masked_softmax_confidence_out = softmax_featurizer_confidence_out[low_confidence_mask]
            del featurizer_confidence_out

            masked_featurizer_click_time_out = featurizer_click_time_out[low_confidence_mask]
            del featurizer_click_time_out

            masked_featurizer_whale_embedding_out = featurizer_whale_embedding_out[low_confidence_mask]
            del featurizer_whale_embedding_out

            transformer_input = torch.cat((masked_featurizer_confidence_out, masked_softmax_confidence_out, masked_featurizer_click_time_out, masked_featurizer_whale_embedding_out), dim=-1)
            transformer_input = transformer_input[:40]
            original_length = transformer_input.shape[0]
            zeros = torch.zeros((40-transformer_input.shape[0], transformer_input.shape[1])).type(torch.cuda.BoolTensor)
            zeros.cuda()
            transformer_input = torch.cat((transformer_input, zeros))
            transformer_input.cuda()
            transformer_input = transformer_input.unsqueeze(0)

            del masked_featurizer_confidence_out
            del masked_softmax_confidence_out
            del masked_featurizer_click_time_out
            del masked_featurizer_whale_embedding_out

            new_mask = []
            low_confidence_index = 0
            num_true = 0
            for i in range(data_mask.shape[0]):
                if data_mask[i]:
                    if low_confidence_mask[low_confidence_index] and num_true < 40:
                        new_mask.append(True)
                        num_true += 1
                    else:
                        new_mask.append(False)
                    low_confidence_index += 1
                else:
                    new_mask.append(False)

            combined_mask = torch.tensor(new_mask).type(torch.cuda.BoolTensor)
            ones = torch.ones((40-num_true,)).type(torch.cuda.BoolTensor)
            ones.cuda()
            zeros = torch.zeros((3040-(40-num_true)-combined_mask.shape[0],)).type(torch.cuda.BoolTensor)
            zeros.cuda()
            combined_mask.cuda()

            combined_mask = torch.cat((combined_mask, ones, zeros), dim=0)

            combined_mask = combined_mask.unsqueeze(0)

            if transformer_input_tensor is None:
                transformer_input_tensor = transformer_input
                original_lengths = [original_length] 
                combined_mask_tensor = combined_mask
                combined_masked_confidence = masked_confidence[:40]
                combined_masked_click_time = masked_click_time[:40]
                combined_masked_whale_number = masked_whale_number[:40]
            else:
                transformer_input_tensor = torch.cat((transformer_input_tensor, transformer_input), dim=0)
                original_lengths.append(original_length)
                combined_mask_tensor = torch.cat((combined_mask_tensor, combined_mask), dim=0)
                combined_masked_confidence = torch.cat((combined_masked_confidence, masked_confidence[:40]), dim=0)
                combined_masked_click_time = torch.cat((combined_masked_click_time, masked_click_time[:40]), dim=0)
                combined_masked_whale_number = torch.cat((combined_masked_whale_number, masked_whale_number[:40]), dim=0)

            del transformer_input

            if (i_batch+1)%samples_per_epoch==0:

                transformer_out = transformer_model(transformer_input_tensor, combined_mask_tensor)
                del transformer_input_tensor, combined_mask_tensor

                combined_transformer_out = None
                for i, length in enumerate(original_lengths):
                    if combined_transformer_out is not None:
                        combined_transformer_out = torch.cat((combined_transformer_out, transformer_out[i][:length]))
                    else:
                        combined_transformer_out = transformer_out[i][:length]

                confidence_out = confidence_model(combined_transformer_out)
                click_time_out = click_time_model(combined_transformer_out)
                whale_embedding_out = whale_embedding_model(combined_transformer_out)
                del combined_transformer_out

                # print('Soundnet time: {}, Featurizer time: {}, Transformer time: {}'.format(soundnet_time, featurizer_time, transformer_time))
                # print(f'After transformer, Total Allocated: {torch.cuda.memory_allocated(0)/(1024*1024)} MiB')

                # Then compute loss for the transformer
                confidence_loss = None
                if confidence_out.shape[0] != 0:
                    confidence_loss = (criterion_confidence(confidence_out, combined_masked_confidence)*40)/samples_per_epoch
                    confidence_out = confidence_out.cpu().data.numpy()
                    confidence_out = np.argmax(confidence_out,axis=1)
                    combined_masked_confidence = list(combined_masked_confidence.cpu().data.numpy())
                    acc_temp = 0
                    for i in range(confidence_out.shape[0]):
                        if combined_masked_confidence[i] == confidence_out[i]:
                            acc_temp += 1
                    acc_list.append(acc_temp/confidence_out.shape[0])
                del confidence_out

                click_time_loss = None
                mask = combined_masked_whale_number.ge(1)
                click_time_out = click_time_out.flatten()
                click_time_out_masked = torch.masked_select(click_time_out, mask)
                click_time_masked = torch.masked_select(combined_masked_click_time, mask)
                if click_time_masked.shape[0] != 0:
                    click_time_loss = (criterion_click_time(click_time_out_masked, click_time_masked)/3)/samples_per_epoch
                    click_time_out = click_time_out_masked.cpu().data.numpy()
                    click_time = click_time_masked.cpu().data.numpy()
                    error_temp = 0
                    total_clicks = 0
                    for i in range(click_time_out.shape[0]):
                        if combined_masked_confidence[i] == 1:
                            error_temp += abs(click_time_out[i] - click_time[i])
                            total_clicks += 1
                    if total_clicks == 0:
                        error_list.append(0)
                    else:
                        error_list.append(error_temp/total_clicks)
                del click_time_out
                del click_time_out_masked
                del click_time_masked
                del combined_masked_confidence

                original_lengths = [0] + original_lengths
                whale_embedding_loss = None
                for i in range(1, len(original_lengths)):
                    start, end = sum(original_lengths[:i]), sum(original_lengths[:i+1])
                    actual_whale_number = combined_masked_whale_number[start:end]
                    actual_whale_number = actual_whale_number[mask[start:end]]
                    actual_whale_embedding_out = whale_embedding_out[start:end]
                    actual_whale_embedding_out = actual_whale_embedding_out[mask[start:end]]
                    if actual_whale_embedding_out.shape[0] > 1:
                        if not whale_embedding_loss:
                            whale_embedding_loss = (criterion_whale_embedding(actual_whale_embedding_out.unsqueeze(1), actual_whale_number)/4)/samples_per_epoch
                        else:
                            whale_embedding_loss += (criterion_whale_embedding(actual_whale_embedding_out.unsqueeze(1), actual_whale_number)/4)/samples_per_epoch
                del actual_whale_embedding_out
                del whale_embedding_out    

                if click_time_loss:
                    avg_val_loss = confidence_loss.data.item() + click_time_loss.data.item()
                    if whale_embedding_loss:
                        avg_val_loss += whale_embedding_loss.data.item()
                        avg_val_loss_whale_embedding.append(whale_embedding_loss.data.item())
                    avg_val_loss_click_time.append(click_time_loss.data.item())
                    avg_val_loss_all.append(avg_val_loss)
                    avg_val_loss_confidence.append(confidence_loss.data.item())

                    del avg_val_loss
                    del confidence_loss
                    del click_time_loss
                    del whale_embedding_loss

                transformer_input_tensor = None

        val_loss = np.mean(avg_val_loss_all)
        print('   Average val loss:', val_loss)
        val_writer.add_scalar('data/loss', val_loss, chkptno)
        losses['val_all'].append(val_loss)
        losses['val_confidence'].append(np.mean(avg_val_loss_confidence))
        losses['val_click_time'].append(np.mean(avg_val_loss_click_time))
        losses['val_whale_embedding'].append(np.mean(avg_val_loss_whale_embedding))

        if chkptno == 0:
            min_val_loss = val_loss

        best = val_loss <= min_val_loss

        if val_loss <= min_val_loss:
            min_val_loss = val_loss

        save_checkpoint({
            'epoch': chkptno,
            'transformer_model' : transformer_model.state_dict(),
            'confidence_model' : confidence_model.state_dict(),
            'click_time_model' : click_time_model.state_dict(),
            'whale_embedding_model' : whale_embedding_model.state_dict(),
        }, chkptno, best)

        acc = np.mean(acc_list)
        error = np.mean(error_list)

        accuracy['val_confidences'].append(acc)
        print("   Accuracy val confidences:", acc)

        accuracy['val_click_times'].append(error)
        print("   Error val click times:", error)

        chkptno = chkptno+1
        
    ########################  TESTING LOOP ############################

    print("\nLoading in best checkpoint...")

    # Load in best checkpoint model
    checkpoint = torch.load(os.path.join(save_folder, 'ckpts', args.exp, 'checkpoint.pth.tar'))

    transformer = ViT(unit_size=24, dim=1024, depth=6, heads=8, mlp_dim=2048, dim_head = 64, dropout = 0.1, emb_dropout = 0.1)
    transformer.cuda()
    transformer = nn.DataParallel(transformer)
    transformer.load_state_dict(checkpoint['transformer_model'])
    transformer.eval()

    confidence_model = confidence_net_transformer()
    confidence_model.cuda()
    confidence_model = nn.DataParallel(confidence_model)
    confidence_model.load_state_dict(checkpoint['confidence_model'])
    confidence_model.eval()

    click_time_model = click_time_net_transformer()
    click_time_model.cuda()
    click_time_model = nn.DataParallel(click_time_model)
    click_time_model.load_state_dict(checkpoint['click_time_model'])
    click_time_model.eval()

    whale_embedding_model = whale_embedding_net_transformer(size=whale_embedding_size)
    whale_embedding_model.cuda()
    whale_embedding_model = nn.DataParallel(whale_embedding_model)
    whale_embedding_model.load_state_dict(checkpoint['whale_embedding_model'])
    whale_embedding_model.eval()

    acc_list = []
    error_list = []

    transformer_input_tensor = None

    print("Now running on test set")
    
    for i_batch, sample_batched in enumerate(test_dataloader):
        
        audio = sample_batched[0][0].type(torch.cuda.FloatTensor) 
        confidence = sample_batched[1][0].type(torch.cuda.LongTensor) 
        click_time = sample_batched[2][0].type(torch.cuda.FloatTensor) 
        whale_number = sample_batched[3][0].type(torch.cuda.LongTensor) 
        data_mask = sample_batched[4][0].type(torch.cuda.BoolTensor)

        # print(f'After loading in data, Total Allocated: {torch.cuda.memory_allocated(0)/(1024*1024)} MiB')

        soundnet_out = soundnet_model(audio)
        soundnet_out = soundnet_out.unsqueeze(1)
        featurizer_1_output = probability_model(probability_soundnet(audio))
        featurizer_1_output = featurizer_1_output.unsqueeze(2)
        combined_input = torch.flatten(torch.matmul(featurizer_1_output, soundnet_out), start_dim=1)
        del soundnet_out
        del featurizer_1_output
        del audio

        mlp_out = mlp_model(combined_input)
        featurizer_confidence_out = featurizer_confidence_model(mlp_out)
        featurizer_click_time_out = featurizer_click_time_model(mlp_out)
        featurizer_whale_embedding_out = featurizer_whale_embedding_model(mlp_out)
        del combined_input
        del mlp_out

        # conversation_length = min(featurizer_confidence_out.shape[0], 4000)

        # print(f'After featurizer, Total Allocated: {torch.cuda.memory_allocated(0)/(1024*1024)} MiB')

        # Then need to filter out all of the low confidence windows
        softmax_featurizer_confidence_out = nn.functional.softmax(featurizer_confidence_out, dim=1)[:,1]
        softmax_featurizer_confidence_out = softmax_featurizer_confidence_out.unsqueeze(1)
        low_confidence_mask = softmax_featurizer_confidence_out.ge(0.1).flatten()  # TODO: Figure out if this is a good enough threshold

        masked_confidence = confidence[low_confidence_mask]
        masked_click_time = click_time[low_confidence_mask]
        masked_whale_number = whale_number[low_confidence_mask]
        del confidence
        del click_time
        del whale_number

        masked_featurizer_confidence_out = featurizer_confidence_out[low_confidence_mask]
        masked_softmax_confidence_out = softmax_featurizer_confidence_out[low_confidence_mask]
        del featurizer_confidence_out

        masked_featurizer_click_time_out = featurizer_click_time_out[low_confidence_mask]
        del featurizer_click_time_out

        masked_featurizer_whale_embedding_out = featurizer_whale_embedding_out[low_confidence_mask]
        del featurizer_whale_embedding_out

        transformer_input = torch.cat((masked_featurizer_confidence_out, masked_softmax_confidence_out, masked_featurizer_click_time_out, masked_featurizer_whale_embedding_out), dim=-1)
        transformer_input = transformer_input[:40]
        original_length = transformer_input.shape[0]
        zeros = torch.zeros((40-transformer_input.shape[0], transformer_input.shape[1])).type(torch.cuda.BoolTensor)
        zeros.cuda()
        transformer_input = torch.cat((transformer_input, zeros))
        transformer_input.cuda()
        transformer_input = transformer_input.unsqueeze(0)

        del masked_featurizer_confidence_out
        del masked_softmax_confidence_out
        del masked_featurizer_click_time_out
        del masked_featurizer_whale_embedding_out

        new_mask = []
        low_confidence_index = 0
        num_true = 0
        for i in range(data_mask.shape[0]):
            if data_mask[i]:
                if low_confidence_mask[low_confidence_index] and num_true < 40:
                    new_mask.append(True)
                    num_true += 1
                else:
                    new_mask.append(False)
                low_confidence_index += 1
            else:
                new_mask.append(False)

        combined_mask = torch.tensor(new_mask).type(torch.cuda.BoolTensor)
        ones = torch.ones((40-num_true,)).type(torch.cuda.BoolTensor)
        ones.cuda()
        zeros = torch.zeros((3040-(40-num_true)-combined_mask.shape[0],)).type(torch.cuda.BoolTensor)
        zeros.cuda()
        combined_mask.cuda()

        combined_mask = torch.cat((combined_mask, ones, zeros), dim=0)

        combined_mask = combined_mask.unsqueeze(0)

        if transformer_input_tensor is None:
            transformer_input_tensor = transformer_input
            original_lengths = [original_length] 
            combined_mask_tensor = combined_mask
            combined_masked_confidence = masked_confidence[:40]
            combined_masked_click_time = masked_click_time[:40]
            combined_masked_whale_number = masked_whale_number[:40]
        else:
            transformer_input_tensor = torch.cat((transformer_input_tensor, transformer_input), dim=0)
            original_lengths.append(original_length)
            combined_mask_tensor = torch.cat((combined_mask_tensor, combined_mask), dim=0)
            combined_masked_confidence = torch.cat((combined_masked_confidence, masked_confidence[:40]), dim=0)
            combined_masked_click_time = torch.cat((combined_masked_click_time, masked_click_time[:40]), dim=0)
            combined_masked_whale_number = torch.cat((combined_masked_whale_number, masked_whale_number[:40]), dim=0)

        del transformer_input

        if (i_batch+1)%samples_per_epoch==0:

            transformer_out = transformer_model(transformer_input_tensor, combined_mask_tensor)
            del transformer_input_tensor, combined_mask_tensor

            combined_transformer_out = None
            for i, length in enumerate(original_lengths):
                if combined_transformer_out is not None:
                    combined_transformer_out = torch.cat((combined_transformer_out, transformer_out[i][:length]))
                else:
                    combined_transformer_out = transformer_out[i][:length]

            confidence_out = confidence_model(combined_transformer_out)
            click_time_out = click_time_model(combined_transformer_out)
            whale_embedding_out = whale_embedding_model(combined_transformer_out)
            del combined_transformer_out

            # print('Soundnet time: {}, Featurizer time: {}, Transformer time: {}'.format(soundnet_time, featurizer_time, transformer_time))
            # print(f'After transformer, Total Allocated: {torch.cuda.memory_allocated(0)/(1024*1024)} MiB')

            # Then compute loss for the transformer
            if confidence_out.shape[0] != 0:
                confidence_out = confidence_out.cpu().data.numpy()
                confidence_out = np.argmax(confidence_out,axis=1)
                combined_masked_confidence = list(combined_masked_confidence.cpu().data.numpy())
                acc_temp = 0
                for i in range(confidence_out.shape[0]):
                    if combined_masked_confidence[i] == confidence_out[i]:
                        acc_temp += 1
                acc_list.append(acc_temp/confidence_out.shape[0])
            del confidence_out

            mask = combined_masked_whale_number.ge(1)
            click_time_out = click_time_out.flatten()
            click_time_out_masked = torch.masked_select(click_time_out, mask)
            click_time_masked = torch.masked_select(combined_masked_click_time, mask)
            if click_time_masked.shape[0] != 0:
                click_time_out = click_time_out_masked.cpu().data.numpy()
                click_time = click_time_masked.cpu().data.numpy()
                error_temp = 0
                total_clicks = 0
                for i in range(click_time_out.shape[0]):
                    if combined_masked_confidence[i] == 1:
                        error_temp += abs(click_time_out[i] - click_time[i])
                        total_clicks += 1
                if total_clicks == 0:
                    error_list.append(0)
                else:
                    error_list.append(error_temp/total_clicks)
            del click_time_out
            del click_time_out_masked
            del click_time_masked
            del combined_masked_confidence

            transformer_input_tensor = None
  

    acc = np.mean(acc_list)
    error = np.mean(error_list)

    print("   Accuracy test confidences:", acc)
    print("   Error test click times:", error)

    plot_losses_and_accuracy(losses, accuracy, plot_save_folder)
