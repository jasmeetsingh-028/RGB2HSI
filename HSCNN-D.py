import torch       #dependencies for building model in pytorch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d 



class Model(nn.Module):

    def __init__(self, input_channels=3, output_channels=31):

        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size = 3, padding = "same")
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size = 1, padding = "same")
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size = 1, padding = "same")
        self.conv4 = torch.nn.Conv2d(64, 16, kernel_size = 3, padding = "same")
        self.conv5 = torch.nn.Conv2d(16, 8, kernel_size = 1, padding = "same")
        self.conv6 = torch.nn.Conv2d(48, 16 , kernel_size = 1, padding = "same")
        self.conv7 = torch.nn.Conv2d(80, 64 , kernel_size = 1, padding = "same")
        self.conv8 = torch.nn.Conv2d(64, 31 , kernel_size = 1, padding = "same")
        


    def forward(self, x):
        
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv1(x))
        x3 = F.relu(self.conv2(x1))
        x4 = F.relu(self.conv2(x2))

        conc1 = torch.concat((x1,x2,x3,x4), dim=1)   #used for concat  #dim=64

        for i in range(38):

            x = F.relu(self.conv3(conc1))

            x5 = F.relu(self.conv4(x))
            x6 = F.relu(self.conv4(x))
            x7 = F.relu(self.conv5(x5))
            x8 = F.relu(self.conv5(x6))

            conc2 = torch.concat((x5,x6,x7,x8), dim=1)  #dim= 48

            x9 = F.relu(self.conv6(conc2))

            conc1 = torch.concat((conc1, x9), dim=1)
            
            conc1 = F.relu(self.conv7(conc1))

            #print("iteration completed: ", i)

            
        output = conc1
        
        output = F.relu(self.conv8(output))

        return output


def set_cuda():

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def main():

    print('imports completed')

    model=Model()
    device= set_cuda()
    input_tensor = torch.randn(1, 3, 482, 512)

    print(f'Input tensor shape: {input_tensor.shape}')
    print(f'Output shape after feeding the tensor to the model: {model.forward(input_tensor).shape}')

if __name__=="__main__":

    main()
