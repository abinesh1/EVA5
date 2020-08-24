# S5 Solution

## Problem Statement

Achieve 99.4% accuracy with lesser than 10000 parameters in 15 epochs 

## Result

Highest accuracy: 99.43%

Number of parameters: 8012

Accuracy consistent over the last 4 epochs

## Approach

Code 1:

1. Create a proper model structure
2. Use 1x1, GAP and fully connected layers to bring the number of parameters
3. On training, required accuracy is not observed to be attained.

Code 2: 

1. Introduce Batch Normalization.
2. On training model appears to overfit and needs some sort of regularization.

Code 3:

1. Added dropout as a form of regularization 
2. Added image augmentation in this case image rotation by 7 degrees
3. On training, accuracy higher than previous models. Training accuracy lesser than test accuracy

Code 4: 

1. Added a learning rate scheduler the varies the learning rate while training to fine tune the performance of the model
2.  Accuracy of 99.4% reached at 12th epoch adn consistent later on

## Network Architecture

```c
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 28, 28]              54
              ReLU-2            [-1, 6, 28, 28]               0
       BatchNorm2d-3            [-1, 6, 28, 28]              12
            Conv2d-4           [-1, 12, 28, 28]             648
              ReLU-5           [-1, 12, 28, 28]               0
         MaxPool2d-6           [-1, 12, 14, 14]               0
         Dropout2d-7		  [-1, 12, 14, 14]               0
       BatchNorm2d-8           [-1, 12, 14, 14]              24
            Conv2d-9           [-1, 16, 14, 14]           1,728
             ReLU-10           [-1, 16, 14, 14]               0
      BatchNorm2d-11           [-1, 16, 14, 14]              32
           Conv2d-12           [-1, 16, 14, 14]           2,304
             ReLU-13           [-1, 16, 14, 14]               0
        MaxPool2d-14             [-1, 16, 7, 7]               0
        Dropout2d-15             [-1, 16, 7, 7]               0
      BatchNorm2d-16             [-1, 16, 7, 7]              32
           Conv2d-17             [-1, 16, 7, 7]           2,304
             ReLU-18             [-1, 16, 7, 7]               0
        AvgPool2d-19             [-1, 16, 1, 1]               0
      BatchNorm2d-20             [-1, 16, 1, 1]              32
           Conv2d-21             [-1, 32, 1, 1]             512
             ReLU-22             [-1, 32, 1, 1]               0
           Conv2d-23             [-1, 10, 1, 1]             330
          Flatten-24                   [-1, 10]               0
       LogSoftmax-25                   [-1, 10]               0
================================================================
Total params: 8,012
Trainable params: 8,012
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.46
Params size (MB): 0.03
Estimated Total Size (MB): 0.46
----------------------------------------------------------------
```

## Code Structure

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, padding=1, bias = False),     # In -  28x28x1, Out -  28x28x6, RF - 3x3
                                   nn.ReLU(),
                                   nn.BatchNorm2d(6),
                                   nn.Conv2d(6, 12, 3, padding = 1, bias = False),  # In -  28x28x6, Out -  28x28x12, RF - 5x5
                                   nn.ReLU(),) 
        
        self.transition1 = nn.Sequential(nn.MaxPool2d(2,2),)                        # In -  28x28x12, Out -  14x14x12, RF - 6x6

        self.conv2 = nn.Sequential(nn.Dropout2d(0.15),
                                   nn.BatchNorm2d(12),
                                   nn.Conv2d(12, 16, 3, padding = 1, bias = False), # In -  14x14x12, Out -  14x14x16, RF - 10x10
                                   nn.ReLU(),
                                   nn.BatchNorm2d(16),
                                   nn.Conv2d(16, 16, 3, padding = 1, bias = False), # In -  14x14x16, Out -  14x14x16, RF - 14x14
                                   nn.ReLU(),)
        
        self.transition2 = nn.Sequential(nn.MaxPool2d(2,2),)                        # In -  14x14x16, Out -  7x7x16, RF - 16x16

        self.conv3 = nn.Sequential(nn.Dropout2d(0.2),
                                   nn.BatchNorm2d(16),
                                   nn.Conv2d(16, 16, 3, padding=1, bias = False),   # In -  7x7x16, Out -  7x7x16, RF - 24x24
                                     nn.ReLU(),)
        
        self.final_layers = nn.Sequential(nn.AvgPool2d(7, 7),
                                          nn.BatchNorm2d(16),
                                          nn.Conv2d(16, 32, 1, bias=False),         # In -  7x7x16, Out -  1x1x16, RF - 48x48
                                          nn.ReLU(),
                                          nn.Conv2d(32, 10, 1),
                                          nn.Flatten(),
                                          nn.LogSoftmax())

                                          
 def forward(self, x):
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.transition2(x)
        x = self.conv3(x)
        x = self.final_layers(x)
        return x                                          
```

## Receptive Field

### 
|         | Kernel size | Padding | Stride | Jout | Nout | RF   |
| ------- | ------ | ------- | ------ | ---- | ---- | ---- |
| input   |        |         |        | 1    | 28   | 1    |
| conv    | 3      | 1       | 1      | 1    | 28   | 3    |
| conv    | 3      | 1       | 1      | 1    | 28   | 5    |
| maxpool | 2      | 0       | 2      | 2    | 14   | 6    |
| conv    | 3      | 1       | 1      | 2    | 14   | 10   |
| conv    | 3      | 1       | 1      | 2    | 14   | 14   |
| maxpool | 2      | 0       | 2      | 4    | 7    | 16   |
| conv    | 3      | 1       | 1      | 4    | 7    | 24   |
| avgpool | 7      | 0       | 7      | 28   | 1    | 48   |
| conv    | 1      | 0       | 1      | 28   | 1    | 48   |
| conv    | 1      | 0       | 1      | 28   | 1    | 48   |