import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_times, F1=8, D=2, dropout_rate=0.5):
        super(  EEGNet  ,self).__init__()
        F2 = F1 * D
        
        self.block1 =nn.Sequential(
            #Temporal Convolution (Learning Frequency)
            nn.Conv2d( 1 , F1 , (1,64) , padding = (0, 32) , bias = False), 
            nn.BatchNorm2d(F1),
            #Depthwise Convolution (Learning Space)
            nn.Conv2d(F1 , F1*D , (n_channels , 1) , group = F1 , bias= False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(), 
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
            
        )
        
        self.block2 = nn.Sequential(
        #Step 3: Separable Convolution (Summarizing)
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Final Classification Head
        self.classifier = nn.Linear(F2 * (n_times // 32), n_classes)
        
        
        
    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
