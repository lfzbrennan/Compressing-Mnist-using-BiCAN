import torch
import torchvision as vision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import mnist_compression


comp_file = "mnist_comp_gan_1.tar"
decomp_file = "mnist_decomp_gan_1.tar"

comp_state_dict = torch.load(comp_file, map_location=torch.device('cpu'))
decom_state_dict = torch.load(decomp_file, map_location=torch.device('cpu'))



c_dims = 200
scalar = 200

compressor = mnist_compression.Compressor(c_dims, scalar=scalar)
decompressor = mnist_compression.Decompressor(c_dims, scalar=scalar)

compressor.load_state_dict(comp_state_dict['state_dict'])
decompressor.load_state_dict(decom_state_dict['state_dict'])

image_size = 32


testing_images_count = 16


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


test_loader = torch.utils.data.DataLoader(
    testing_images,
    batch_size=testing_images_count,
    shuffle=True
)

for image_batch in test_loader:

    images = image_batch[0]

    compressed_images = compressor(images)

    decompressed_images = decompressor(compressed_images)


    break


plt.figure(figsize=(4, 4))
plt.axis("off")
plt.title("Original Images")
plt.imshow(np.transpose(vutils.make_grid(images.detach(), padding=2, normalize=True), (1, 2, 0)))

plt.show()

plt.figure(figsize=(4, 4))
plt.axis("off")
plt.title("Decompressed Images")
plt.imshow(np.transpose(vutils.make_grid(decompressed_images.detach(), padding=2, normalize=True), (1, 2, 0)))

plt.show()

print("success")
