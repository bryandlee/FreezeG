# FreezeG
Freezing generator for \*pseudo\* image translation


Inspired by the training footage of [FreezeD](https://github.com/sangwoomo/FreezeD) trasfer learning, I have tested a simple idea of freezing the early layers of the generator in transfer learning settings, and it worked pretty well. Reusing the high-level layers of a pre-trained generator for image-to-image translation is not a novel idea [[1]](https://papers.nips.cc/paper/7480-one-shot-unsupervised-cross-domain-translation.pdf), [[2]](https://arxiv.org/pdf/2007.13332.pdf), and I guess it could be also applied to the transfer learning of noise-to-image GANs. This is a *pseudo* translation method because the input image should be projected to the learned latent space first, and then the projected vector is propagated again to generate the target image. Therefore, the performance is limited to the in-domain images of the original GAN. I used [StyleGAN2 implementation](https://github.com/rosinality/stylegan2-pytorch), and below are some of the results I've got. By also fixing the latent vector of the early layers and manipulating the ones that are fed into the last layers, the rendering style can be controlled separately. For the datasets with large geometric transformations such as face2simpsons, the connection between the original image and the resulting image becomes less intuitive. See [cat2flower](https://github.com/bryandlee/FreezeG#cat2flower-afhq-oxfordflowers) for an extreme case. 



### Cat2Wild [AFHQ]
<img src="./imgs/cat2wild/3.gif" width="320"> &nbsp; <img src="./imgs/cat2wild/1.gif" width="320"> &nbsp;\
<img src="./imgs/cat2wild/2.gif" width="650"> &nbsp; 



### Face2Malnyun [FFHQ, Malnyun]
<img src="./imgs/face2malnyun/1.png" width="320"> &nbsp; <img src="./imgs/face2malnyun/2.png" width="320"> &nbsp;\
<img src="./imgs/face2malnyun/3.png" width="320"> &nbsp; <img src="./imgs/face2malnyun/4.png" width="320"> &nbsp;\
<img src="./imgs/face2malnyun/5.png" width="320"> &nbsp; <img src="./imgs/face2malnyun/6.png" width="320"> &nbsp;\
<img src="./imgs/face2malnyun/7.png" width="320"> &nbsp; <img src="./imgs/face2malnyun/8.png" width="320"> &nbsp;


Interpolation

<img src="./imgs/face2malnyun/gif_1.gif" width="320"> &nbsp; <img src="./imgs/face2malnyun/gif_2.gif" width="320"> &nbsp;\
<img src="./imgs/face2malnyun/gif_3.gif" width="320"> &nbsp; <img src="./imgs/face2malnyun/gif_4.gif" width="320"> &nbsp;\
<img src="./imgs/face2malnyun/gif_5.gif" width="320"> &nbsp; <img src="./imgs/face2malnyun/gif_6.gif" width="320"> &nbsp;\
<img src="./imgs/face2malnyun/gif_7.gif" width="320"> &nbsp; <img src="./imgs/face2malnyun/gif_8.gif" width="320"> &nbsp;\
<img src="./imgs/face2malnyun/gif_9.gif" width="320"> &nbsp; <img src="./imgs/face2malnyun/gif_10.gif" width="320"> &nbsp;


Failures

<img src="./imgs/face2malnyun/fail_1.png" width="320"> &nbsp; <img src="./imgs/face2malnyun/fail_2.png" width="320"> &nbsp;


### Face2Simpsons [FFHQ, Simpsons]
<img src="./imgs/face2simpsons/1.png" width="320"> &nbsp; <img src="./imgs/face2simpsons/2.png" width="320"> &nbsp;\
<img src="./imgs/face2simpsons/3.png" width="320"> &nbsp; <img src="./imgs/face2simpsons/4.png" width="320"> &nbsp;\
<img src="./imgs/face2simpsons/5.png" width="320"> &nbsp; <img src="./imgs/face2simpsons/6.png" width="320"> &nbsp;\
<img src="./imgs/face2simpsons/7.png" width="320"> &nbsp; <img src="./imgs/face2simpsons/8.png" width="320"> &nbsp;\
<img src="./imgs/face2simpsons/9.png" width="320"> &nbsp; <img src="./imgs/face2simpsons/10.png" width="320"> &nbsp;\
<img src="./imgs/face2simpsons/11.png" width="320"> &nbsp; <img src="./imgs/face2simpsons/12.png" width="320"> &nbsp;\
<img src="./imgs/face2simpsons/13.png" width="320"> &nbsp; <img src="./imgs/face2simpsons/14.png" width="320"> &nbsp;\
<img src="./imgs/face2simpsons/15.png" width="320"> &nbsp; <img src="./imgs/face2simpsons/16.png" width="320"> &nbsp;\
<img src="./imgs/face2simpsons/17.png" width="320"> &nbsp; <img src="./imgs/face2simpsons/18.png" width="320"> &nbsp;


Interpolation

<img src="./imgs/face2simpsons/gif_1.gif" width="320"> &nbsp; <img src="./imgs/face2simpsons/gif_2.gif" width="320"> &nbsp;\
<img src="./imgs/face2simpsons/gif_3.gif" width="320"> &nbsp; <img src="./imgs/face2simpsons/gif_4.gif" width="320"> &nbsp;

Failures

<img src="./imgs/face2simpsons/fail_1.png" width="320"> &nbsp; <img src="./imgs/face2simpsons/fail_2.png" width="320"> &nbsp;\
<img src="./imgs/face2simpsons/fail_3.png" width="320"> &nbsp; <img src="./imgs/face2simpsons/fail_4.png" width="320"> &nbsp;\
<img src="./imgs/face2simpsons/fail_5.png" width="320"> &nbsp; <img src="./imgs/face2simpsons/fail_6.png" width="320"> &nbsp;\
<img src="./imgs/face2simpsons/fail_7.png" width="320"> &nbsp; <img src="./imgs/face2simpsons/fail_8.png" width="320"> &nbsp;



### Face2Dog [FFHQ, AFHQ]
<img src="./imgs/face2dog/1.png" width="320"> &nbsp; <img src="./imgs/face2dog/2.png" width="320"> &nbsp;\
<img src="./imgs/face2dog/3.png" width="320"> &nbsp; <img src="./imgs/face2dog/4.png" width="320"> &nbsp;\
<img src="./imgs/face2dog/5.png" width="320"> &nbsp; <img src="./imgs/face2dog/6.png" width="320"> &nbsp;\
<img src="./imgs/face2dog/7.png" width="320"> &nbsp; <img src="./imgs/face2dog/8.png" width="320"> &nbsp;\
<img src="./imgs/face2dog/9.png" width="320"> &nbsp; <img src="./imgs/face2dog/10.png" width="320"> &nbsp;\
<img src="./imgs/face2dog/11.png" width="320"> &nbsp; <img src="./imgs/face2dog/12.png" width="320"> &nbsp;\
<img src="./imgs/face2dog/13.png" width="320"> &nbsp; <img src="./imgs/face2dog/14.png" width="320"> &nbsp;\
<img src="./imgs/face2dog/15.png" width="320"> &nbsp; <img src="./imgs/face2dog/16.png" width="320"> &nbsp;\
<img src="./imgs/face2dog/17.png" width="320"> &nbsp; <img src="./imgs/face2dog/18.png" width="320"> &nbsp;


Interpolation

<img src="./imgs/face2dog/gif_1.gif" width="320"> &nbsp; <img src="./imgs/face2dog/gif_2.gif" width="320"> &nbsp;\
<img src="./imgs/face2dog/gif_3.gif" width="320"> &nbsp; <img src="./imgs/face2dog/gif_4.gif" width="320"> &nbsp;

Failures

<img src="./imgs/face2dog/fail_1.png" width="320"> &nbsp; <img src="./imgs/face2dog/fail_2.png" width="320"> &nbsp;\
<img src="./imgs/face2dog/fail_3.png" width="320"> &nbsp; <img src="./imgs/face2dog/fail_4.png" width="320"> &nbsp;



### Face2Art [FFHQ, MetFaces]
<img src="./imgs/face2art/2.gif" width="320"> &nbsp; <img src="./imgs/face2art/3.gif" width="320"> &nbsp;\
<img src="./imgs/face2art/1.gif" width="650"> &nbsp; 



### Cat2Flower [AFHQ, OxfordFlowers]
<img src="./imgs/cat2flower/1.gif" width="320"> &nbsp; <img src="./imgs/cat2flower/2.gif" width="320"> &nbsp;

