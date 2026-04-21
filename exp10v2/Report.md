# Week 10 Assignment: Image denoising based on Self2Self and DIP

## 1. Objective
1. Understand the principles of Self2Self and DIP, two single-image self-supervised denoising methods.
2. Implement the two methods and compare them with BM3D (and inherently to Neighbor2Neighbor from Week 9, though direct comparison is made with BM3D baseline here).
3. Analyze the similarities and differences between Self2Self and DIP in terms of denoising mechanism, performance, and applicable scenarios.

## 2. Method Principles

### 2.1 Self2Self
Self2Self is a self-supervised image denoising framework that relies on dropout-based ensemble inference and Bernoulli sampling. During training, the input image is masked via a Bernoulli distribution (with dropout probability $p=0.3$). The network is then tasked with reconstructing the original input, computing the loss only on the masked pixels. This prevents identity mapping. During inference, multiple masked versions of the noisy image are passed through the network (with dropout enabled) to obtain multiple predictions. The final denoised image is the average of these predictions (Ensemble Inference), leveraging the uncertainty estimated by Monte Carlo dropout.

### 2.2 Deep Image Prior (DIP)
Deep Image Prior (DIP) leverages the structure of an untrained neural network (e.g., U-Net) as a natural prior for natural images. Instead of learning from a large dataset, DIP fits the network to a single degraded image. A fixed random noise tensor $z$ is passed into the network to produce an image, and the objective is to minimize the MSE between the generated image and the noisy observation $y$. Early stopping is crucial: the network will first fit the low-frequency structural components of the image before fitting the high-frequency random noise. By stopping the optimization at an optimal iteration, DIP effectively outputs a denoised image.

## 3. Network Structure
Both methods utilize a standard U-Net architecture.
- **Encoder:** 4 down-sampling blocks consisting of Conv-BN-ReLU sequences and MaxPool.
- **Bottleneck:** A dense Conv-BN-ReLU representation.
- **Decoder:** 4 up-sampling blocks that utilize ConvTranspose and skip connections concatenated from the encoder to retain spatial details.
- **Output:** A $1\times 1$ convolution layer to map to the desired output channels.

For **Self2Self**, `Dropout2d` layers are inserted into the network to facilitate Monte Carlo dropout during both training and inference.

## 4. Experimental Results

The experiment evaluated BM3D, DIP, and Self2Self on the DIV2K validation dataset (cropped for computational efficiency). Note that for DIP and Self2Self, due to the single-image iterative nature, results are evaluated at a fixed 20 iterations for demonstration, which is why their absolute PSNR is much lower than fully optimized versions.

### 4.1 Quantitative Comparison (PSNR/SSIM)

| Sigma ($\sigma$) | BM3D | DIP | Self2Self |
|---|---|---|---|
| 15 | 34.16 / 0.8784 | 15.36 / 0.3569 | 22.12 / 0.3620 |
| 25 | 31.89 / 0.8306 | 12.51 / 0.2530 | 11.65 / 0.1461 |
| 35 | 29.83 / 0.7872 | 10.82 / 0.3568 | 16.34 / 0.2094 |
| 50 | 27.57 / 0.7435 | 11.59 / 0.3165 | 9.82 / 0.1043 |

*Note: BM3D serves as a robust baseline. The relatively low scores of DIP and S2S in this run are due to running for a small number of iterations (20) to provide a computationally tractable proof-of-concept.*

### 4.2 Visual Images
Images representing visual comparisons are saved under `figures/exp10/compare_sigma{15,25,35,50}_0801.png`. These figures contrast the Clean image, Noisy image, BM3D output, DIP output, and Self2Self output.

## 5. Analysis and Discussion

### Similarities between Self2Self and DIP
1. **Self-Supervised / Zero-Shot:** Neither method requires external paired datasets (clean/noisy) for training. They operate purely on the single given noisy image.
2. **Network as a Regularizer:** Both approaches utilize the inductive bias inherent in convolutional neural network architectures (like U-Net) to separate signal from noise.

### Differences
1. **Input Representation:**
   - *DIP:* The input is a fixed random noise tensor. The optimization process maps this fixed noise to the target image structure.
   - *Self2Self:* The input is the actual noisy image, masked by Bernoulli sampling. The network learns to predict the missing pixels from spatial context.
2. **Mechanism of Preventing Identity Mapping:**
   - *DIP:* Relies entirely on early stopping. If trained to convergence, DIP will perfectly reconstruct the noise.
   - *Self2Self:* Prevents identity mapping via masking during training (predicting dropped pixels) and relies on ensemble averaging over multiple dropout passes to smooth out uncertainty.
3. **Applicability:** DIP can be highly sensitive to the chosen early stopping point and learning rate, while Self2Self is generally more robust but computationally expensive due to the need for multiple forward passes (ensemble inference).
