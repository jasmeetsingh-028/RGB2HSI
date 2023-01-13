import torch       #dependencies for building model in pytorch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.utils.data import DataLoader  


class Model(nn.Module):

    def __init__(self, input_channels=3, output_channels=31):

        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, 8, kernel_size = 3, padding = "same")
        self.conv2 = torch.nn.Conv2d(8, 8, kernel_size = 1, padding = "same")
        self.conv3 = torch.nn.Conv2d(8, 16, kernel_size = 3, padding = "same")
        self.conv4 = torch.nn.Conv2d(16, 16, kernel_size = 1, padding = "same")
        self.conv5 = torch.nn.Conv2d(16, 32, kernel_size = 3, padding = "same")

        self.conv6 = torch.nn.Conv2d(32, 32, kernel_size = 1, padding = "same")   #concat

        self.conv7 = torch.nn.Conv2d(32, 64, kernel_size = 3, padding = "same")

        self.conv8 = torch.nn.Conv2d(64, 64, kernel_size = 1, padding = "same")    #concat

        self.conv9 = torch.nn.Conv2d(64, 128, kernel_size = 3, padding = "same")
        self.conv10 = torch.nn.Conv2d(128, 128, kernel_size = 1, padding = "same")
        self.conv11 = torch.nn.Conv2d(128, 64, kernel_size = 3, padding = "same")

        self.conv12 = torch.nn.Conv2d(64, 64, kernel_size = 1, padding = "same")  #concat

        self.conv13 = torch.nn.Conv2d(128, 64, kernel_size = 3, padding = "same")
        self.conv14 = torch.nn.Conv2d(64, 32, kernel_size = 3, padding = "same")

        self.conv15 = torch.nn.Conv2d(32, 32, kernel_size = 1, padding = "same")  #concat

        self.conv16 = torch.nn.Conv2d(64, 31, kernel_size = 3, padding = "same")
        self.conv17 = torch.nn.Conv2d(31, output_channels, kernel_size = 1, padding = "same")

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        conc11 = F.relu(self.conv6(x))  #concat dim-32-32

        x = F.relu(self.conv7(conc11))

        conc21 = F.relu(self.conv8(x))  #concat

        x = F.relu(self.conv9(conc21)) 
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))

        conc22 = F.relu(self.conv12(x))

        x = torch.concat((conc21,conc22), dim=1)  #concatination layer dim - 128

        x = F.relu(self.conv13(x)) #dim - 128-64
        x = F.relu(self.conv14(x)) #dim - 64-32

        conc12 = F.relu(self.conv15(x)) #dim - 32-32
        
        x = torch.concat((conc12,conc11), dim=1)  #concatination layer dim- 64

        x = F.relu(self.conv16(x))
        output = self.conv17(x)

        return output
       
def set_cuda():

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device

     
def main():
    set_cuda()

    model=Model()

    device= set_cuda()

    print(device)

    #model.to(device)

    input_tensor = torch.randn(1, 3, 482, 512)

    print(f'Input tensor shape: {input_tensor.shape}')

    print(f'Output shape after feeding the tensor to the model: {model.forward(input_tensor).shape}')

if __name__ == "__main__":
    main()