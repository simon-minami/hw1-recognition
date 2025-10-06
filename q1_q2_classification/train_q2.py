import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import argparse
import random
from datetime import datetime


class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        ##################################################################
        # TODO: Define a FC layer here to process the features
        ##################################################################
        for param in self.resnet.parameters():
            param.requires_grad = False
            # freeze all layers
        # replace last layer with right number of output classes
        self.resnet.fc =  nn.Linear(512, num_classes)


        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        

    def forward(self, x):
        ##################################################################
        # TODO: Return raw outputs here
        ##################################################################
        x = self.resnet(x)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    ##################################################################
    # TODO: Create hyperparameter argument class
    # We will use a size of 224x224 for the rest of the questions. 
    # Note that you might have to change the augmentations
    # You should experiment and choose the correct hyperparameters
    # You should get a map of around 50 in 50 epochs
    ##################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=5, type=int, help='epochs')
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('-g', '--gamma', default=1.0, type=float, help='gamma, (how much to reduce lr)')
    parser.add_argument('-s', '--step_size', default=1, type=int, help='for learning rate scheduling')
    parser.add_argument('-notes', '--experiment_notes', default=None, type=str, help='for tensorboard log dir')


    a = parser.parse_args()
    args = ARGS(
        epochs=a.epochs,
        inp_size=224,
        use_cuda=torch.cuda.is_available(),
        val_every=200,
        lr=a.learning_rate,
        batch_size=a.batch_size,
        step_size=a.step_size,
        gamma=a.gamma
    )
    experiment_name = f"{a.experiment_notes}_lr{a.learning_rate}_b{a.batch_size}_ss{a.step_size}_g{a.gamma}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    
    print(args)

    ##################################################################
    # TODO: Define a ResNet-18 model (https://arxiv.org/pdf/1512.03385.pdf) 
    # Initialize this model with ImageNet pre-trained weights
    # (except the last layer). You are free to use torchvision.models 
    ##################################################################

    model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler, experiment_name)
    print('test map:', test_map)
