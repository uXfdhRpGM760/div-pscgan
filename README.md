
# Image denoising with diversity enhanced posterior sampling conditional GAN

![teaserFigure](https://raw.githubusercontent.com/uXfdhRpGM760/div-pscgan/main/figure_2j.png)

Image denoisng is an inverse problem with many possible solutions. Despite enormous progress, popular denoising methods aim at a single best solution and struggle with a number of limitations. Supervised Convolutional Neural Networks tend to produce blurry images, denoising Generative Adversarial Network (GAN) models suffer from mode collapse issues, while variational autoencoders so far underperform on complex data sets. These limitations affect quantitative applications, such as scientific image denoising, where generation of representative distributions of denoised images is of value. The posterior sampling conditional GAN model is a recent development that produces state of the art samples of denoised images with high peak signal-to-noise ratio and low Fr\'echet Inception Distance. However, the sampled images are visually similar. Here we introduce a variance enhanced version of the posterior sampling conditional GAN model, aiming to generate broader distributions of denoised images. The method incorporates an additional diversity loss term, which steers the conditional GAN model training. We provide denoising results and discuss their quality for simple and more complex data sets, compare denoised image distributions to the ones obtained with existing approaches, and show that the presented simple method offers competitive image denoising  while sampling a broader set of solutions.
### Information

This repository hosts the code for the publication "Image denoising with diversity enhanced posterior sampling conditional GAN
" 

### Dependencies 
To install all dependencies, follow the following steps.

You could use the same virtual environment (`div-pscgan`) as used by us by following the steps below.
 
1. Clone the repository locally by using the command `git clone https://github.com/uXfdhRpGM760/div-pscgan.git`
2. Move into the cloned folder by using the command `cd div-pscgan`. 
3. Create a new environment by entering the python command in the terminal `conda env create -f div-pscgan.yml`. This will install all dependencies needed to run the notebooks.
4. Then run the command `conda activate div-pscgan`.

### Getting Started
You can start by looking in the `examples` directory and try out the notebooks.