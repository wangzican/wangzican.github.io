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
type: 'ICCV 2025'
thumbnail: '/images/higherorder/thumb.png'
code_url: 'https://github.com/wangzican/Stochastic-Gradient-Estimation-for-Higher-order-Differentiable-Rendering#'
pdf: 'https://arxiv.org/abs/2412.03489'
---

<body>
<div class="globaldiv">

<div class="grey-box" style="max-width: 800px; margin: 0 auto; padding: 20px;">
<br>
    <p style="margin: 0 auto; text-align: center;">
    <span style="font-size: 24px;"><b>Stochastic Gradient Estimation for Higher-order Differentiable Rendering</b></span> <br><br>
    <span style="font-size: 17px; color: black">ICCV 2025</span><br><br>
    <span style="font-size: 17px;"><a class="hiddenlink" href="https://wangzican.github.io/">Zican Wang<sup>1</sup></a>, <a class="hiddenlink" href="https://mfischer-ucl.github.io/">Michael Fischer<sup>1,2</sup></a>, <a class="hiddenlink" href="https://www.homepages.ucl.ac.uk/~ucactri/">Tobias Ritschel<sup>1</sup></a></span><br>
    <a style="font-size: 14px;" class="hiddenlink" href="https://www.ucl.ac.uk/"><sup>1</sup>University College London</a>, <a style="font-size: 14px;" class="hiddenlink" href="https://research.adobe.com/"><sup>2</sup>Adobe Research</a>
</p>
<br>
</div>
<div class="row" style="margin: 50px 0 50px 0">
    <div style="display: inline">
        <ul style="list-style: none; text-align: center">
            <li class="horizItem">
                <a href="https://arxiv.org/pdf/2412.03489" download="Higher_order_differentiable_rendering.pdf" class="mylink">
                <img class="teaserbutton" src="/images/higherorder/front_page.png" ><br>
                    <h4><strong>Paper</strong></h4>
                </a>
            </li>
            <li class="horizItem">
                <a href="files/hodr/ICCV_2025_Stochastic_Estimation_for_Higher_order_Differentiable_Rendering_Part2.pdf" download="Higher_order_differentiable_rendering_supplementary.pdf" class="mylink">
                <img class="teaserbutton" src="/images/icons/clip.png" ><br>
                    <h4><strong>Supplemental</strong></h4>
                </a>
            </li>
            <li class="horizItem">
                <a href="https://github.com/wangzican/Stochastic-Gradient-Estimation-for-Higher-order-Differentiable-Rendering#" class="mylink">
                <img class="teaserbutton" src="/images/icons/github_logo.svg" ><br>
                    <h4><strong>Code</strong></h4>
                </a>
            </li>
            <li class="horizItem">
                <a href="files/hodr/v4.pdf" download="ICCV2025_poster.pdf" class="mylink">
                <img class="teaserbutton" src="/images/icons/poster.png" ><br>
                    <h4><strong>Poster</strong></h4>
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

<p style="text-align: center;">
 <img src="/images/higherorder/concept.png" width="100%" margin-top="50px" />
</p>

<b>Interactive Demo</b><br>

<p style="text-align: justify">
Below is an interactive 1D example which uses our higher order method to differentiate through a discontinuous step function. The task here is to move the triangle center (parameterized by theta), such that it covers the grey pixel at the bottom. The plateaus in the cost landscape come from the fact that the error between the pixel's desired color and its current color does not take into account how "far away" the triangle is when it's not overlapping the pixel. We can smoothen these plateaus by our proposed convolution with a Gaussian kernel (displayed in plot in the top left corner, click 'Show Smooth' to see the convolved function). 
We then sample this convoluted space to estimate a second order gradient (Hessian) and use the samples to drive a second order optimizer that moves the initial parameter (blue) towards the region of zero cost, i.e., such that the triangle overlaps the pixel. This is compared to the original first order method that is shown in red. <br>
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
<b>Derivation and sampling</b>
<br>
<p style="text-align: justify">
The Hessian of a multivariate Gaussian function describes the second-order partial derivatives of that function. For sampling a diagonal Gaussian Hessian, we can use inverse transform sampling via the Smirnov transform. To allow sampling with this method, the 2nd order derivative is positivized and normalized in to a probability density function. Since the curvature of the Gaussian is the derivative of its gradient, we can get the CDF by scaling and combining the gradient of the Gaussian. 

For a off-diagonal entry \( H_{ij}(x) \) in the Hessian matrix is given by:
\[
H_{ij}(x) = \frac{\partial^2 f}{\partial x_i \partial x_j}
\]
Because the multivariate Gaussian is separable, each Hessian component decomposes as follows:
<br>
- The second derivative of the 1D Gaussian in \(x_i\) and \(x_j\)
<br>
- Times the Gaussian (unchanged) in the other dimensions.
</p>
<br>
<figure>
  <img src="/images/higherorder/DetailDerivation.png" 
       alt="Detailed derivation" 
       height="200">
</figure>

<br>

<b>Results</b>
<br>
<p style="text-align: justify">
Quantitative results of different methods on different tasks (rows) and their convergence plots. We report convergence time in wall-clock units, in ratio to the overall best method, OurHVPA. In the numerical columns, .9 and .99 report the time taken to achieve 90 and 99% error reduction from the initial starting configuration, respectively, while the bar plots graphically show these findings. The line plots report image- and parameter-space convergence in the left and right column, respectively, on a log-log scale.
Methods:
</p>
<br>
<figure>
  <img src="/images/higherorder/table_test.png" alt="table_test" width="100%">
</figure>
<br>

<div class="image-row">
  <figure>
    <img src="/images/500x300.png" alt="banana" width="200">
    <figcaption>First order</figcaption>
  </figure>
  <figure>
    <img src="/images/higherorder/banana.gif" alt="banana" width="200">
    <figcaption>OurHVP</figcaption>
  </figure>
  <figure>
    <img src="/images/higherorder/bananaGT.png" alt="bananaGT" width="200">
    <figcaption>Ground truth</figcaption>
  </figure>
</div>
<br>
<div class="image-row">
  <figure>
    <img src="/images/500x300.png" alt="suzanne" width="200">
    <figcaption>First order</figcaption>
  </figure>
  <figure>
    <img src="/images/higherorder/suzanne.gif" alt="suzanne" width="200">
    <figcaption>OurHVP</figcaption>
  </figure>
  <figure>
    <img src="/images/higherorder/SuzanneGT.jpg" alt="suzanneGT" width="200">
    <figcaption>Ground truth</figcaption>
  </figure>
</div>




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


