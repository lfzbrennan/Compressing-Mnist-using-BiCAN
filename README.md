# Compressing MNIST with BiCAN
 Like most networks that operate on an MNIST dataset, this implementation of a Bicycle Convolutional Adverserial Network is being used as a proof of concept of using a BiCAN in this way: compression. A BiCAN works by using two networks in unision, a compressor and decompressor. These networks work much in the same way as a DCGAN works. The compressor is a convolutional network that tries to compress the entire image into as few bytes as possible, while the decompressor works to turn those bytes back into the same image using a transposed convolutional net setup.
 
 This type of network requires two cycle loss: one for the cycled image, and one for the cycled compressed image. Like most cycle losses, this loss is implemented by using the mean squared error: in this case, the error between the original image /decompressed image, and the compressed image/decompressed compressed image. The idea behind using both cycle losses is that we want to equally prioritize an effective compressing, and decompressing. Much like other adverserial nets, if either network gets too good at doing what its doing, training effectively stops. We want to minimize this happening as much as possible - using both the decompressor and compressor loss in tandem works to try to fix this problem. 
 
**Network Steps**
1. Compress original image using compressor.
1. Decompress the result to form a new recreated image.
1. Compress the result of the recreated image.

**Losses**
* Decompressor Loss: MSE loss between original image and recreated image (step 2)
* Compressor Loss: MSE loss between original compressed image (step 1) and compressed recreated image (step 3)

**Results**

Using the most effective model, each mnist image was able to be compressed to 200 bytes, with a loss of roughly 1%. After compiling them into plain text, npy, and gz files, I compared the file sizes to the original mnist dataset. These comparisons are shown below.

Also a kinda cool side note - I ran all these models on a machine-learning server I built out of the skeleton of an old gaming pc.

<img src="https://github.com/lfzbrennan/Compressing-Mnist-using-BiCAN/blob/master/supplementary_images/data_table.png">

**Direct Comparisons**

* .npy.gz: 2.9 Mb -> 1.9 Mb: Size reduced 53%
* .npy: 10.2 Mb -> 2.0 Mb: Size reduced **410%**
* .txt.gz: 9.9 Mb -> 2.5 Mb: Size reduced 296%
* .txt: 47 Mb -> 7.4 Mb: Size reduced **535%**

Overall, I was pretty impressed with these results. With only 1% loss, I was able to compress the entire mnist dataset to less than 1/5 of its original size. 


