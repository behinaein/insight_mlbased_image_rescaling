import cv2
import numpy as np
import torch
#import tqdm
#from torch.utils.data import DataLoader

#from torch.optim import Adam

from dfnet.model import DFNet
from dfnet.loss import InpaintLoss
import torch.optim as optim
from torch.optim import lr_scheduler
#from torchvision import datasets, models, transforms
import time
import os
import copy
#import matplotlib.pyplot as plt


class Trainer:
    """
    class to train the DFNet
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.init_model(model_path)

    @property
    def input_size(self):
        """
        for future use in cases that we have different input sizes
        :return: model input size
        """
        return (512, 512)

    def init_model(self, path):
        """
        initialize the class
        :param path: path the the model
        :return:
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using gpu.')
        else:
            self.device = torch.device('cpu')
            print('Using cpu.')

        self.model = DFNet().to(self.device)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        print('Model %s loaded.' % path)

    def get_name(self, path):
        """
        strips the path and return the file name
        :param path: file with full path
        :return: file name
        """
        return '.'.join(path.name.split('.')[:-1])

    def run_model(self, image_path):
        """
        run the model on the image provided
        :param image_path: full path tho the image
        :return: None
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (512, 512)) # resize to the CNN input size
        mask = self.create_mask()
        #change the image and mask dimension and type to be used in CNN
        img = img.reshape(1, *img.shape).astype(np.uint8)
        mask = mask.reshape(1, *mask.shape, 1).astype(np.uint8)

        return self.inpaint(img, mask)


    def inpaint(self, imgs, masks):
        """
        performs the inpaint on the images using the mask provided.
        :param imgs: input images in format of (#samples, width, height, #channels)
        :param masks: masks for images
        :return: reconstructed images
        """

        imgs = np.transpose(imgs, [0, 3, 1, 2])
        masks = np.transpose(masks, [0, 3, 1, 2])

        imgs = torch.from_numpy(imgs).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)

        # scale images
        imgs = imgs.float().div(255)
        masks = masks.float().div(255)

        # apply the mask on images
        imgs_miss = imgs * masks

        results = self.model(imgs_miss, masks)

        return results[0], imgs, masks

    def optimizer_SGD(self):
        """
        optimizer of stochastic gradient descent
        :return: an SGD optimizer
        """
        return  optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def scheduler(self, optimizer_ft):
        """
        create a scheduler for changing the learning rate of the optimizer
        :param optimizer_ft: an optimizer
        :return: a scheduler
        """
        return lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    def train_model(self, data_dir, custom_loss, optimizer, scheduler, num_epochs=25):
        """
        trains a deep fusion network
        :param data_dir: directory for images and masks for training and validation
        :param custom_loss: loss function
        :param optimizer: an optimizer
        :param scheduler:  a learning rate scheduler
        :param num_epochs: number of epochs to train the model
        :return: the best model weights
        """
        since = time.time() # start time the training phase

        # stores the best model so far
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e18

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                # if phase == 'train':
                #     model.train()  # Set model to training mode
                # else:
                #     model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                files = os.listdir(os.path.join(data_dir, phase))
                for input_file_name in files:
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    input_file = os.path.join(data_dir, phase, input_file_name)
                    with torch.set_grad_enabled(phase == 'train'):
                        results, img, mask = self.run_model(input_file)
                        loss, _ = custom_loss(results, img, mask)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * 1

                if phase == 'train':
                    scheduler.step()  # change the learning rate

                epoch_loss = running_loss

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since  # compute the training time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_loss))

        # return the  best model weights
        return self.best_model_wts

    def create_mask(self):
        """
        create a random mask of the size of CNN input
        :return: a mask
        """
        buttom = np.random.randint(400, 512)
        left = np.random.randint(0, 100)
        right = np.random.randint(400, 512)
        top = np.random.randint(0, 100)

        mask = np.zeros((512, 512))
        # change the color of the sides to black in order to mask the input image in those areas
        mask[0:top, :] = 255
        mask[buttom:512, :] = 255
        mask[:, 0:left] = 255
        mask[:, right:512] = 255
        
        return mask

if __name__ == '__main__':
    
    data_dir = 'datasets'
    model_file = 'model/model_places2.pth'
    
    # if GPUs are available, use them
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using gpu.')
    else:
        device = torch.device('cpu')
        print('Using cpu.')

    custom_loss = InpaintLoss().to(device)
    trainer = Trainer(model_file)

    optimizer_ft = trainer.optimizer_SGD()
    exp_lr_scheduler = trainer.scheduler(optimizer_ft)
    trainer.train_model(data_dir, custom_loss, optimizer_ft, exp_lr_scheduler, num_epochs=25)

    
