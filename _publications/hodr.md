---
# page settings
permalink: /publication/hodr
classes: wide
author_profile: false
layout: mylayout # default is single in _config.yml

# Info
title: "Stochastic Gradient Estimation for Higher-order Differentiable Rendering"
show_title: false
authors:
  - name: "Zican Wang"
    # url: "https://wangzican.github.io/"
  - name: "Michael Fischer"
    url: "https://mfischer-ucl.github.io/"
  - name: "Tobias Ritschel"
    url: "https://www.homepages.ucl.ac.uk/~ucactri/"
collection: publications
category: 'Conference Papers'
excerpt: 'Second order optimization for black-box functions such as Path tracing'
date: 2024-11-15
type: 'arXiv preprint'
thumbnail: '/images/higherorder/thumb.png'
pdf: 'https://arxiv.org/abs/2412.03489'
---

<body>
<div class="globaldiv">

<div class="grey-box" style="max-width: 800px; margin: 0 auto; padding: 20px;">
<br>
    <p style="margin: 0 auto; text-align: center;">
    <span style="font-size: 24px;"><b>Stochastic Gradient Estimation for Higher-order Differentiable Rendering</b></span> <br><br>
    <span style="font-size: 17px; color: black">arXiv preprint</span><br><br>
    <span style="font-size: 17px;"><a class="hiddenlink" href="https://wangzican.github.io/">Zican Wang</a>, <a class="hiddenlink" href="https://mfischer-ucl.github.io/">Michael Fischer</a>, <a class="hiddenlink" href="https://www.homepages.ucl.ac.uk/~ucactri/">Tobias Ritschel</a></span><br>
    <a style="font-size: 14px;" class="hiddenlink" href="https://www.ucl.ac.uk/">University College London</a>
</p>
<br>
</div>
<div class="row" style="margin: 50px 0 50px 0">
    <div style="display: inline">
        <ul style="list-style: none; text-align: center">
            <li class="horizItem">
                <a href="/files/Higher_order_differentiable_rendering.pdf" download="Higher_order_differentiable_rendering.pdf">
                <img class="teaserbutton" src="/images/higherorder/front_page.png" ><br>
                    <h4><strong>Paper</strong></h4>
                </a>
            </li>
            <li class="horizItem">
                <a href="/files/Higher_order_differentiable_rendering_supplementary.pdf" download="Higher_order_differentiable_rendering_supplementary.pdf">
                <img class="teaserbutton" src="/images/icons/clip.png" ><br>
                    <h4><strong>Supplemental</strong></h4>
                </a>
            </li>
        </ul>
    </div>
</div>


<b>Abstract</b><br>
<p style="text-align: justify">
We derive methods to compute higher order differentials(Hessians and Hessian-vector products) of the rendering
operator. Our approach is based on importance sampling of a convolution that represents the differentials of rendering parameters and shows to be applicable to both rasterization and path tracing. We further suggest an aggregate sampling strategy to importance-sample multiple dimensions of one convolution kernel simultaneously. We demonstrate that this information improves convergence when used in higher-order optimizers such as Newton or Conjugate Gradient relative to a gradient descent baseline in several inverse rendering tasks.
</p>

<p align="center">
 <img src="/images/higherorder/concept.png" width="100%" margin-top="50px" />
</p>

<b>Interactive Demo</b><br>

<p style="text-align: justify">
Below is an interactive 1D example which uses our method to differentiate through a discontinuous step function. The task here 
is to move the triangle center (parameterized by theta), such that it covers the black pixel at the bottom. The plateaus in the cost landscape 
come from the fact that the error between the pixel's desired and its current color does not take into account how "far away" the triangle is 
when it's not overlapping the pixel. We can smoothen these plateaus by our proposed convolution with a Gaussian kernel (displayed in plot in the right bottom corner, click 'Show Smooth' to see the convolved function). 
We then sample this convoluted space and use the samples to drive a gradient descent that moves the initial 
parameter (green) towards the region of zero cost, i.e., such that the triangle overlaps the pixel. <br>
</p>

<iframe src="/demo/hodr/index.html" width="100%" height="700px" style="border: none;"></iframe>

<!-- 
<br>
<p style="text-align: justify">
We also provide a simple 2D example of our method in <a href="https://colab.research.google.com/github/mfischer-ucl/prdpt/blob/main/examples/box_example.ipynb">Colab</a>. Here, we optimize a square that, in the initial configuration, 
does not overlap its reference and hence creates a plateau in the loss landscape (the 2D counterpart to the example above). This example uses a simpler renderer 
and hence does not need all the scene config / rendering infrastructure used in the main repository.
</p>
<div style="display: flex; justify-content: center; align-items: center; margin-top: 2%; max-width: 100%">
  <img src="/assets/images/prdpt/2Dexample.png" style="max-width: 90%;">
</div> -->

<br>

<b>Results</b><br>


<br>
<!-- 
<b>Citation</b><br>
If you find our work useful and use parts or ideas of our paper or code, please cite: <br>
<p class="cite-box" style="margin-top: 5px">
  <span style="font-family: Lucida Console, Courier New, monospace; padding: 10px;">
    @inproceedings{fischer2023plateau, <br>
      &nbsp;&nbsp;title={Plateau-Reduced Differentiable Path Tracing}, <br> 
      &nbsp;&nbsp;author={Fischer, Michael and Ritschel, Tobias}, <br>
      &nbsp;&nbsp;booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition}, <br>
      &nbsp;&nbsp;pages={4285--4294}, <br>
      &nbsp;&nbsp;year={2023} <br>
    }
  </span>
</p> -->


<p style="text-align: justify">
<b>Acknowledgements</b><br>
Our approach is based on the PRDPT paper by <a href="https://mfischer-ucl.github.io/">Michael Fischer</a> and <a href="https://www.homepages.ucl.ac.uk/~ucactri/">Tobias Ritschel</a>. With additional higher order optimization and sampling scheme. Please check out the original paper <a href="https://mfischer-ucl.github.io/prdpt/">here</a>.</p>
</div>
</body>


