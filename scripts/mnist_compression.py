import numpy as np
import torch.utils.data.dataset as dataset
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import torchvision as vision
from torch.nn.modules import flatten
#import matplotlib.pyplot as plt

import torchvision.utils as vutils


# custom flatten layer

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# custom expand layer

class Expand(nn.Module):
    def forward(self, input):
        return input.view(-1, 200*4, 4, 4)


class Compressor(nn.Module):

    def __init__(self, c_dims, scalar=128):
        super(Compressor, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=scalar, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(in_channels=scalar, out_channels=scalar*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(scalar * 2),

            nn.Conv2d(in_channels=scalar*2, out_channels=scalar*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(scalar * 4),

            Flatten(),

            nn.Linear(in_features=scalar*4*4*4, out_features=scalar*4*4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(scalar*4*4),

            nn.Linear(in_features=scalar*4*4, out_features=c_dims),
            nn.Tanh(),

        )


    def forward(self, i):

        return self.main(i)


class Decompressor(nn.Module):

    def __init__(self, c_dims, scalar=128):
        super(Decompressor, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(in_features=c_dims, out_features=scalar*4*4),
            nn.ReLU(True),
            nn.BatchNorm1d(scalar * 4 * 4),

            nn.Linear(in_features=scalar*4*4, out_features=scalar*4*4*4),
            nn.LeakyReLU(True),

            Expand(),

            nn.ConvTranspose2d(in_channels=scalar*4, out_channels=scalar*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(in_channels=scalar * 2, out_channels=scalar, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(in_channels=scalar, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()

        )

        return

    def forward(self, i):

        return self.main(i)


def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if str(device) == 'cuda:0':
        print("GPU running...")


    image_size = 32 # image size set at (32, 32) instead of (28, 28) for ease of model and scalability

    # load datasets


    print("Loading Data...")
    train_set = vision.datasets.MNIST(
        root='.data/Mnist',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([.5], [.5])
        ])
    )

    test_set = vision.datasets.MNIST(
        root='.data/Mnist',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([.5], [.5])
        ])
    )

    # set batch sizes/sampling frequencies

    batch_size = 100
    sampling_frequency = 100  # batches

    train_set_size = len(train_set)

    val_batch_size = 100
    val_batch_num = int(len(test_set)/ val_batch_size)


    print("Device: " + str(device))

    # create datasets


    data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=True
    )

    # compressed image size

    c_dims = 200

    # set amount of parameters

    scalar = 200

    # start epochs at 0

    epoch = 0

    #set total epochs

    total_epochs = 20

    print("Creating Models and Optimizers...")

    compressor = Compressor(c_dims, scalar=scalar).to(device)
    decompressor = Decompressor(c_dims, scalar=scalar).to(device)

    # set learning rates

    learning_rate_com = .0002
    learning_rate_decom = .0002

    # build optimizers and losses

    com_opt = opt.Adam(compressor.parameters(), lr=learning_rate_com)
    decom_opt = opt.Adam(decompressor.parameters(), lr=learning_rate_decom, weight_decay=1e-4)

    decompressed_loss = nn.MSELoss()
    compressed_loss = nn.MSELoss()

    torch.cuda.empty_cache()

    decompressed_loss_list = []
    compressed_loss_list = []

    loss_in_a_row_count = 3
    loss_count = 0

    prev_decompressor_loss = 0
    prev_compressor_loss = 0

    current_decompressor_loss = 0
    current_compressor_loss = 0

    print("Starting Training...")


    if (str(device) == 'cuda:0'):
        torch.cuda.empty_cache()
    for i in range(total_epochs):

        count = 0

        epoch += 1

        perc = 50

        for batch in data_loader:

            if (str(device) == 'cuda:0'):
                torch.cuda.empty_cache()

            count += 1

            com_opt.zero_grad()
            decom_opt.zero_grad()

            images = batch[0].to(device)

            # cycle the compressed/decompressed images

            compressed_images = compressor(images).to(device)

            decompressed_images = decompressor(compressed_images).to(device)

            cycled_compressed_images = compressor(decompressed_images).to(device)

            # calculate cycled losses

            decom_loss = decompressed_loss(images, decompressed_images)
            com_loss = compressed_loss(compressed_images, cycled_compressed_images)

            decom_loss.backward(retain_graph=True)
            com_loss.backward()

            com_opt.step()
            decom_opt.step()

            decom_loss = decom_loss.item()
            com_loss = com_loss.item()

            # print results

            if count % sampling_frequency == 0:
                print("Epoch: " + str(epoch) + "    " + "Batch #: " + str(count) + "/" + str(int(train_set_size/batch_size))+ "    " + "Decompressor Loss: " +
                    str(decom_loss.__round__(4)) + "    Compressor Loss: " + str(com_loss.__round__(4)))

        if (str(device) == 'cuda:0'):
            torch.cuda.empty_cache()

        current_compressor_loss = 0
        current_decompressor_loss = 0

        # validation set for epoch

        for val_batch in test_loader:

            if (str(device) == 'cuda:0'):
                torch.cuda.empty_cache()

            com_opt.zero_grad()
            decom_opt.zero_grad()

            images = val_batch[0].to(device)

            compressed_images = compressor(images).to(device)

            decompressed_images = decompressor(compressed_images).to(device)

            cycled_compressed_images = compressor(decompressed_images).to(device)

            decom_loss = decompressed_loss(images, decompressed_images)
            com_loss = compressed_loss(compressed_images, cycled_compressed_images)

            decom_loss = decom_loss.item()
            com_loss = com_loss.item()

            current_compressor_loss += com_loss
            current_decompressor_loss += decom_loss

            if epoch == 1:
                prev_compressor_loss = com_loss
                prev_decompressor_loss = decom_loss


        # find total epoch losses

        current_compressor_loss /= val_batch_num
        current_decompressor_loss /= val_batch_num

        # add epoch losses to list

        decompressed_loss_list.append(current_decompressor_loss)
        compressed_loss_list.append(current_compressor_loss)

        # printing out results

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        print("Epoch " + str(epoch) + " Final::-->  Decompressor Loss: " + str(current_decompressor_loss)
                + "    Compressor Loss: " + str(current_compressor_loss))

        print("-----------------------------------------------------------")


        # check if training has reached a maximum

        if current_compressor_loss > prev_compressor_loss and current_decompressor_loss > prev_decompressor_loss:
            loss_count += 1
        else:
            loss_count = 0
        if loss_count > loss_in_a_row_count:
            print("Losses have been going up -->> model will be force stopped")
            break
        if current_compressor_loss == 0.:
            print("Compressor loss is zero -->> model cannot train and will be force stopped")

        prev_decompressor_loss = current_decompressor_loss
        prev_compressor_loss = current_compressor_loss


    print("Model Done")


    for i in range(epoch):
        print("Epoch: " + str(i + 1) + "    Decompressor Loss: " + str(decompressed_loss_list[i]) + "   Compressor Loss: " + str(compressed_loss_list[i]))


    comp_state= {
        'state_dict': compressor.state_dict()
    }

    decomp_state = {
        'state_dict': decompressor.state_dict()
    }


    torch.save(comp_state, "mnist_comp_gan_1.tar")
    torch.save(decomp_state, 'mnist_decomp_gan_1.tar')

    cols = 4
    rows = 4

    sample_images = []

    decompressed_sample_images = []

    for sample_batch in data_loader:
        sample_images = sample_batch[0].to(device)

        compressed_sample_images = compressor(sample_images).to(device)

        decompressed_sample_images = decompressor(compressed_sample_images).to(device)

        break
'''
    plt.figure(figsize=(10, 5))
    plt.axis("off")
    plt.title("Decompressed Images")
    plt.imshow(np.transpose(vutils.make_grid(decompressed_sample_images.to(device).detach(), padding=2, normalize=True), (1, 2, 0)))

    plt.show()

    epoch_level = []

    for i in range(len(decompressed_loss_list)):
        epoch_level.append(i+1)

    plt.scatter(epoch_level, decompressed_loss_list, marker='o')
    plt.scatter(epoch_level, compressed_loss_list, marker='s')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.show()
'''
if __name__ == "__main__":
    main()




