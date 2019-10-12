import torch
import torchvision as vision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import gzip
import mnist_compression

def create_file(c_dims=200, scalar=200, ctype='o', encoding='npy', compressed_file=False, comp_file="mnist_comp_gan_1.tar", decomp_file="mnist_decomp_gan_1.tar", file_name="before_training_images"):


    if (ctype == 'c' or ctype == 'd'):

        comp_state_dict = torch.load(comp_file, map_location=torch.device('cpu'))
        compressor = mnist_compression.Compressor(c_dims, scalar=scalar)
        compressor.load_state_dict(comp_state_dict['state_dict'])

        if (ctype == 'd'):
            decomp_state_dict = torch.load(decomp_file, map_location=torch.device('cpu'))
            decompressor = mnist_compression.Decompressor(c_dims, scalar=scalar)
            decompressor.load_state_dict(decomp_state_dict['state_dict'])

    image_size = 32
    testing_images = vision.datasets.MNIST(
        root='.data/Mnist',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([.5], [.5])
        ])
    )


    batch_size = len(testing_images)

    test_loader = torch.utils.data.DataLoader(
        testing_images,
        batch_size=batch_size
    )


    for image_batch in test_loader:

        print("Loading images...")

        images = image_batch[0]

        if (ctype == 'c' or ctype == 'd'):

            print("Compressing images...")

            compressed_images = compressor(images)
            if (ctype == 'd'):

                print("Decompressing images...")

                decompressed_images = decompressor(compressed_images)
                detached_images = decompressed_images.detach().numpy()
            else:
                detached_images = compressed_images.detach().numpy()

        else:
            detached_images = images.detach().numpy()

        print("Processing...")

        detached_images /= 2.0
        detached_images +=.5
        detached_images *= 256

        detached_images = detached_images.astype(np.ubyte)

        detached_images = detached_images.flatten()

    print("Creating file...")

    file_name_processed = file_name

    if(encoding=='npy'):
        file_name_processed += '.npy'
        
    if (compressed_file):
        file_name_processed += '.gz'
        file_name_processed = gzip.GzipFile(file_name_processed, "w")

    if (encoding=='npy'):
        np.save(file=file_name_processed, arr=detached_images)
    else:
        np.savetxt(file_name_processed, detached_images, encoding='utf-8', fmt='%s')

    if(compressed_file):
        file_name_processed.close()

    print("done creating file")

# type: 'o', 'c', 'd'
# encoding: 'npy', 'txt'

c_dims=200
scalar=200

create_file(c_dims=c_dims, scalar=scalar, compressed_file=True, encoding='txt', ctype='c')

print("success")