import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import transforms, utils, datasets
from tqdm import tqdm
from torch.nn.parameter import Parameter
import pdb
import torchvision
import os
import gzip
import tarfile
import gc   # gc is garbage collector
from IPython.core.ultratb import AutoFormattedTB



class CancerDataset(Dataset):
    def __init__(self, root, download=True, size=256, train=True):
        if download and not os.path.exists(os.path.join(root, 'cancer_data')):
            datasets.utils.download_url('http://liftothers.org/cancer_data.tar.gz', root, 'cancer_data.tar.gz', None)
            self.extract_gzip(os.path.join(root, 'cancer_data.tar.gz'))
            self.extract_tar(os.path.join(root, 'cancer_data.tar'))
        
        postfix = 'train' if train else 'test'
        root = os.path.join(root, 'cancer_data', 'cancer_data')
        self.dataset_folder = torchvision.datasets.ImageFolder(os.path.join(root, 'inputs_' + postfix) ,transform = transforms.Compose([transforms.Resize(size),transforms.ToTensor()]))
        self.label_folder = torchvision.datasets.ImageFolder(os.path.join(root, 'outputs_' + postfix) ,transform = transforms.Compose([transforms.Resize(size),transforms.ToTensor()]))

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)
  
    @staticmethod
    def extract_tar(tar_path):
        print('Untarring {}'.format(tar_path))
        z = tarfile.TarFile(tar_path)
        z.extractall(tar_path.replace('.tar', ''))

    def __getitem__(self,index):
        img = self.dataset_folder[index]
        label = self.label_folder[index]
        return img[0],label[0][0]
  
    def __len__(self):
        return len(self.dataset_folder)



class Unet(nn.Module):
    """
    Custom U-net architecture. 
    Based on architecture from paper (https://arxiv.org/pdf/1505.04597.pdf)

    Modified to work with dataset images.
    """
    
    def __init__(self, dataset): # Unet constructor
        # You always need to use super to call nn.Module's init function when creating a class that inherits nn.Module.
        super(Unet, self).__init__()
        
        # Always initialize your layers in the init function.
        # - Lump 1
        self.L1conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L1relu2 = nn.ReLU(inplace=True)
        self.L1conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L1relu4 = nn.ReLU(inplace=True)
        self.L1maxp5 = nn.MaxPool2d(kernel_size=(2,2))
    
        # - Lump 2
        self.L2conv1 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L2relu2 = nn.ReLU(inplace=True)
        self.L2conv3 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L2relu4 = nn.ReLU(inplace=True)
        self.L2maxp5 = nn.MaxPool2d(kernel_size=(2,2))

        # - Lump 3
        self.L3conv1 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L3relu2 = nn.ReLU(inplace=True)
        self.L3conv3 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L3relu4 = nn.ReLU(inplace=True)
        self.L3maxp5 = nn.MaxPool2d(kernel_size=(2,2))

        # - Lump 4
        self.L4conv1 = nn.Conv2d(256, 512, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L4relu2 = nn.ReLU(inplace=True)
        self.L4conv3 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L4relu4 = nn.ReLU(inplace=True)
        self.L4maxp5 = nn.MaxPool2d(kernel_size=(2,2))

        # - Lump 5 (the bottom of the U)
        self.L5conv1 = nn.Conv2d(512, 1024, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L5relu2 = nn.ReLU(inplace=True)
        self.L5conv3 = nn.Conv2d(1024, 1024, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L5relu4 = nn.ReLU(inplace=True)
        self.L5ucon5 = nn.ConvTranspose2d(1024, 512, kernel_size=(2,2), stride=(2,2))#, padding=(2,2)) #kernel_size=(2,2), stride=1, padding=(1,1))

        # - Lump 6
        self.L6conv1 = nn.Conv2d(1024, 512, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L6relu2 = nn.ReLU(inplace=True)
        self.L6conv3 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L6relu4 = nn.ReLU(inplace=True)
        self.L6ucon5 = nn.ConvTranspose2d(512, 256, kernel_size=(2,2), stride=(2,2)) #, padding=(1,1))

        # - Lump 7
        self.L7conv1 = nn.Conv2d(512, 256, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L7relu2 = nn.ReLU(inplace=True)
        self.L7conv3 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L7relu4 = nn.ReLU(inplace=True)
        self.L7ucon5 = nn.ConvTranspose2d(256, 128, kernel_size=(2,2), stride=(2,2)) #kernel_size=(2,2), stride=1, padding=(1,1))

        # - Lump 8
        self.L8conv1 = nn.Conv2d(256, 128, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L8relu2 = nn.ReLU(inplace=True)
        self.L8conv3 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L8relu4 = nn.ReLU(inplace=True)
        self.L8ucon5 = nn.ConvTranspose2d(128, 64, kernel_size=(2,2), stride=(2,2)) #kernel_size=(2,2), stride=1, padding=(1,1))

        # - Lump 9
        self.L9conv1 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L9relu2 = nn.ReLU(inplace=True)
        self.L9conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=(1,1))
        self.L9relu4 = nn.ReLU(inplace=True)
        self.L9conv5 = nn.Conv2d(64, 2, kernel_size=(1,1), stride=1)#, padding=(1,1))


    def forward(self, input): # nn.Module sets up a hook that calls forward when you "call" the module object: net(x) calls net.forward(x)
    
        # Lump 1
        L1conv1_out = self.L1conv1(input)
        L1relu2_out = self.L1relu2(L1conv1_out)
        L1conv3_out = self.L1conv3(L1relu2_out)
        L1relu4_out = self.L1relu4(L1conv3_out) # Will skip connect to Lump 9
        L1maxp5_out = self.L1maxp5(L1relu4_out)

        # Lump 2
        L2conv1_out = self.L2conv1(L1maxp5_out)
        L2relu2_out = self.L2relu2(L2conv1_out)
        L2conv3_out = self.L2conv3(L2relu2_out)
        L2relu4_out = self.L2relu4(L2conv3_out) # Will skip connect to Lump 8
        L2maxp5_out = self.L2maxp5(L2relu4_out)

        #Lump 3
        L3conv1_out = self.L3conv1(L2maxp5_out)
        L3relu2_out = self.L3relu2(L3conv1_out)
        L3conv3_out = self.L3conv3(L3relu2_out)
        L3relu4_out = self.L3relu4(L3conv3_out) # Will skip connect to Lump 7
        L3maxp5_out = self.L3maxp5(L3relu4_out)

        #Lump 4
        L4conv1_out = self.L4conv1(L3maxp5_out)
        L4relu2_out = self.L4relu2(L4conv1_out)
        L4conv3_out = self.L4conv3(L4relu2_out)
        L4relu4_out = self.L4relu4(L4conv3_out) # Will skip connect to Lump 6
        L4maxp5_out = self.L4maxp5(L4relu4_out)

        #Lump 5
        L5conv1_out = self.L5conv1(L4maxp5_out)
        L5relu2_out = self.L5relu2(L5conv1_out)
        L5conv3_out = self.L5conv3(L5relu2_out)
        L5relu4_out = self.L5relu4(L5conv3_out) 
        L5ucon5_out = self.L5ucon5(L5relu4_out)

        #Lump 6
        L6conv1_out = self.L6conv1(torch.cat((L4relu4_out, L5ucon5_out), dim=1))
        L6relu2_out = self.L6relu2(L6conv1_out)
        L6conv3_out = self.L6conv3(L6relu2_out)
        L6relu4_out = self.L6relu4(L6conv3_out) 
        L6ucon5_out = self.L6ucon5(L6relu4_out)

        #Lump 7
        L7conv1_out = self.L7conv1(torch.cat((L3relu4_out, L6ucon5_out), dim=1))
        L7relu2_out = self.L7relu2(L7conv1_out)
        L7conv3_out = self.L7conv3(L7relu2_out)
        L7relu4_out = self.L7relu4(L7conv3_out) 
        L7ucon5_out = self.L7ucon5(L7relu4_out)

        #Lump 8
        L8conv1_out = self.L8conv1(torch.cat((L2relu4_out, L7ucon5_out), dim=1))
        L8relu2_out = self.L8relu2(L8conv1_out)
        L8conv3_out = self.L8conv3(L8relu2_out)
        L8relu4_out = self.L8relu4(L8conv3_out) 
        L8ucon5_out = self.L8ucon5(L8relu4_out)

        #Lump 9
        L9conv1_out = self.L9conv1(torch.cat((L1relu4_out, L8ucon5_out), dim=1))
        L9relu2_out = self.L9relu2(L9conv1_out)
        L9conv3_out = self.L9conv3(L9relu2_out)
        L9relu4_out = self.L9relu4(L9conv3_out) 
        L9conv5_out = self.L9conv5(L9relu4_out)

        return L9conv5_out

def print_prediction(epoch, batch, ims, ax, val_dataset, model):
    """
    Creates a frame of the animation. 
    A sample image and an overlayed cancer prediction.

    Parameters
        ims (list of lists) : each row is a list of artists to draw 
            in the current frame; here we are just animating one 
            artist, the image, in each frame
    """

    #Pull pos_test_000072  
    data_point = val_dataset[172]
    
    #Grab image and label
    x = data_point[0].unsqueeze(0).cuda()
    y = data_point[1].cuda()

    #Make prediction
    y_pred = model(x)

    #Prep for display
    y_pred = y_pred.squeeze(0).permute(1,2,0).argmax(dim=2)
    # y_pred.detach()
    y_pred = y_pred.cpu()
    y_pred = y_pred.detach().numpy()

    # Extract original image
    input = data_point[0].permute(1,2,0)
    label = data_point[1]
    im1 = ax.imshow(input, cmap="binary", alpha=1.0)

    # Get Cancer Prediction overlay
    ax.set_title(f"Learning to Detect Cancer") # [epoch:{epoch}, batch:{batch}]")
    ax.set_yticks([])
    ax.set_xticks([])
    im2 = ax.imshow(y_pred, cmap="binary", alpha=0.2, animated=True)
    
    # Store for later animation
    ims.append([im1, im2])

    pass


def scope():
  

    
  pass
    


def main():

    # Hyperparameters
    num_epochs = 3

    # Initialize Datasets: 
    #   - list of tuples(3 x 512 x 512 input tensor & 512 x 512 label tensor)
    train_dataset = CancerDataset("/tmp/cancer_dataset", train=True)  #len=1342
    val_dataset = CancerDataset("/tmp/cancer_dataset", train=False)   #len=175

    # Initialize Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=6, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=6, pin_memory=True, shuffle=True)

    # Initialize Model
    model = Unet(train_dataset)
    model.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Loss Function
    objective = nn.CrossEntropyLoss()

    # Lists for saving values
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []


    ## Train the Unet

    # Initialize pieces for animation
    fig, ax = plt.subplots() #figure()
    ims = []

    try:

        # Training Loop
        for epoch in range(num_epochs):

            # For displaying progress
            loop = tqdm(total=len(train_loader), position=0, leave=False)

            for batch, (x, y_truth) in enumerate(train_loader):
                x, y_truth = x.cuda(), y_truth.cuda()

                # Zero the gradient
                optimizer.zero_grad()

                # Predict y
                y_hat = model(x)

                # Check how close the predition is to the truth
                loss = objective(y_hat, y_truth.long()) #y_hat.float(), y_truth.long()

                # Backpropagate the loss through network
                loss.backward()

            # Take a step and update the network parameters
            optimizer.step()

            # Store loss value
            train_loss_list.append(loss.item())

            # Calculate accuracy (Might need to flatten to check)
            train_accuracy_percent = (torch.softmax(y_hat, 1).argmax(1) == y_truth).float().mean()
            train_accuracy_list.append(train_accuracy_percent)

            #Validation
            if (batch % (len(train_loader)-1)//2 == 0) or (batch == 0) or (batch == len(train_loader)-1):

                temp_list = []
                list_of_val_losses = []
                for x, y in val_loader:
                    x, y = x.cuda(), y.cuda()
                    y_hat = model(x)
                    list_of_val_losses.append(objective(y_hat, y.long()).item())
                    best_guess = y_hat.argmax(1)
                    compare = (y==best_guess).float()
                    accuracy = torch.sum(compare).item()/torch.numel(compare)  #Figure out what is supposed to go here (not 16)
                    temp_list.append(accuracy)
            
                mean_val = np.mean(list_of_val_losses)
                val_loss_list.append((len(train_loss_list), mean_val))
                val_accuracy_list.append( (len(train_loss_list), np.mean(np.array(temp_list))) )

            #Check out an image for animation:
            if batch % 1 == 0: #(len(train_loader)-1)/2 == 0:
                print_prediction(epoch, batch, ims, ax, val_dataset, model)

            # Manage memory
            gc.collect()
            memory_update = torch.cuda.memory_allocated() / 1e9

            loop.set_description( f'epoch:{epoch+1} \n loss:{loss.item()}\n training accuracy:{train_accuracy_percent.item()} \n memory:{memory_update} GB' )
            loop.update(True)

        # Create an animation
        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=1000)
        # Save the animation
        ani.save('cancer_detector.mp4')

    except:
        __ITB__()

    pass

if __name__ == "__main__":
    # Check whether we have cuda (This code presupposes we can use cuda.)
    assert torch.cuda.is_available(), "You need a cuda-enabled GPU to run this code"
    
    #Formatting tracebacks for ipython."__" == 'dunder' or 'double underscore'
    __ITB__ = AutoFormattedTB(mode = 'Verbose',color_scheme='LightBg', tb_offset = 1)

    # Do deep learning
    main()