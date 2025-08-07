// Constants
const EPSILON = 1e-4;
const SQRT_2PI = Math.sqrt(2 * Math.PI);


const stepEdge = x => (x < -0.5 || x > 2.5 ? 1 : 0);

const calcGaussian1D = (x, sigma) => {
    const denom = sigma * SQRT_2PI;
    return Math.exp(-x * x / (2 * sigma * sigma)) / denom;
};

const calcGradGaussian1DPDF = (x, sigma) => {
    // already sampling the positivized function here 
    const alpha = 0.5 * sigma * SQRT_2PI;
    const val = (Math.abs(x) / Math.pow(sigma, 2)) * calcGaussian1D(x, sigma = sigma);
    return alpha * val;
};
const calcHessGaussian1DCDF = (x, sigma) => {
    // CDF of the Hessian Gaussian, i.e., the integral of the PDF a reshaped grad gaussian
    const factor = x / (4 * sigma) * Math.exp(0.5);
    const shape = Math.exp(-x * x / (2 * sigma * sigma));
    const val = x < -sigma ? -factor * shape : x > sigma ? 1 - factor * shape : 0.5 + factor * shape;
    return val;
};

const calcHessGaussian1DPDF = (x, sigma) => {
    const denom = Math.pow(sigma, 2);
    const gaussian = calcGaussian1D(x, sigma);
    const val = (Math.pow(x, 2) / Math.pow(sigma, 4)) * gaussian - (1 / denom) * gaussian;
    const beta = sigma ** 2 / 4 * Math.exp(0.5) * SQRT_2PI;
    return beta * Math.abs(val);
};

const gradGaussianKernel = (x, sigma) => {
    return -(x / (sigma * sigma)) * calcGaussian1D(x, sigma);
};

const hessGaussianKernel1D = (x, sigma) => {
    return (Math.pow(x, 2) / Math.pow(sigma, 4)) * calcGaussian1D(x, sigma) - (1 / denom) * calcGaussian1D(x, sigma);
};

const getPDF = (x, sigma) => 0.5 * sigma * SQRT_2PI * x;


// sample from a standard normal, N(0,1)
const boxMuller = () => {
    return Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
};

// convert standard normal samples to scaled normal 
const sampleGaussian = (sigma = 1.0) => sigma * boxMuller();

const icdfGradGauss = (x, sigma) => {
    const term = -2 * sigma * sigma * Math.log(2 * (x > 0.5 ? 1 - x : x));
    return Math.sqrt(term);
};

const cleanRandom = (x) => {
    if (x < EPSILON) { x += EPSILON; }					// too close to zero 
    if (x - 0.5 < EPSILON) { x -= EPSILON; }		// too close to 0.5
    if (0.5 - x < EPSILON) { x += EPSILON; }		// too close to 0.5
    if (1.0 - x < EPSILON) { x -= EPSILON; }		// too close to 1
    return x;
};

function inverseHessCDF(sigma) {
    // Return a function interpolate(y): given cdf interpolate the x from a table of cfd<->x
    // Generate x values from -10*sigma to +10*sigma
    const xRange = Array.from({ length: 2000 }, (_, i) => -10 * sigma + (20 * sigma * i) / 1999);
    // Generate corresponding CDF values
    const yRange = xRange.map(x => calcHessGaussian1DCDF(x, sigma));

    function interpolate(y) {
        if (y <= yRange[0]) return 0;
        if (y >= yRange[yRange.length - 1]) return 1;

        let i = yRange.findIndex(v => v > y);
        if (i === 0) return xRange[0];

        const x0 = xRange[i - 1], x1 = xRange[i];
        const y0 = yRange[i - 1], y1 = yRange[i];
        const t = (y - y0) / (y1 - y0);
        return x0 + t * (x1 - x0);
    }
    return interpolate;
}

const getHessGaussianSamples = (n_samples, sigma, antithetic) => {
    const icdf = inverseHessCDF(sigma);
    const samples = [];
    const f_x = [];
    const p_x = [];
    for (var i = 0; i < n_samples; i++) {
        let rand = Math.random();
        let x_i = icdf(rand);
        samples.push(x_i);
        if (antithetic) {
            let arand = 1.0 - rand;
            let ax = icdf(arand);
            samples.push(ax);
        }
    }
    const beta = sigma ** 2 / 4 * Math.exp(0.5) * SQRT_2PI;
    for (let s of samples) {
        var p_xi = calcHessGaussian1DPDF(s, sigma);
        var f_xi = p_xi / beta;
        f_x.push(f_xi);
        p_x.push(p_xi);
    }
    return [samples, f_x, p_x];
};

const getGradGaussianSamples = (n_samples, sigma, antithetic) => {
    const samples = [];
    const f_x = [];
    const p_x = [];
    for (var i = 0; i < n_samples; i++) {

        let rand = cleanRandom(Math.random());
        let sign = rand < 0.5 ? -1.0 : 1.0;
        let x_i = icdfGradGauss(rand, sigma) * sign;
        samples.push(x_i);

        if (antithetic) {
            let arand = 1.0 - rand;
            sign = arand < 0.5 ? -1.0 : 1.0;
            let ax = icdfGradGauss(arand, sigma) * sign;
            samples.push(ax);
        }
    }
    // calculcate sample value and pdf 
    for (let s of samples) {
        var f_xi = Math.abs(s) / Math.pow(sigma, 2) * calcGaussian1D(s, sigma);
        var p_xi = calcGradGaussian1DPDF(s, sigma);
        f_x.push(f_xi);
        p_x.push(p_xi);
    }

    return [samples, f_x, p_x];
};

const getGaussianSamples = (n_samples, sigma, antithetic) => {
    const samples = [];
    const p_x = [];
    for (let i = 0; i < n_samples; i++) {
        var x_i = sampleGaussian(sigma);
        samples.push(x_i);
        if (antithetic) { samples.push(-x_i); }
    }

    // calc sample value 
    for (s of samples) {
        p_x.push(calcGaussian1D(s, sigma));
    }
    return [samples, p_x];
}

const mcEstimate = (f_x, p_x) => {
    const N = f_x.length;
    let estimate = f_x.reduce((acc, f_xi, i) => acc + (f_xi / p_x[i]), 0.0);
    return estimate / N;
}

const mse = (x, y) => (x - y) ** 2;

const convolve = (theta, n_samples, samples, pdfs, sigma, goal = 0.0) => {
    const outputs = [];
    for (let i = 0; i < n_samples; i += 1) {
        const tau = samples[i];
        const w = gradGaussianKernel(tau, sigma);
        const fn = stepEdge(theta - tau);
        const weighted_fn_val = mse(fn, goal) * w;
        outputs.push(weighted_fn_val);
    }

    var final_estimate = mcEstimate(outputs, pdfs);
    return final_estimate;
};

const avgList = vals => vals.reduce((acc, val) => acc + val, 0) / vals.length;

const optimizeFristOrder = (sigma, theta, stepsize, numSamples, antithetic, gt_theta) => {
    let nsamples_real = antithetic ? Math.round(numSamples / 2.0) : numSamples;
    // get gradient by convolving and multiplying by kernel: 
    const [x_i, f_xi, p_xi] = getGradGaussianSamples(numSamples, sigma, antithetic);
    const grad = convolve(theta, nsamples_real, x_i, p_xi, sigma, gt_theta);
    // GD
    theta -= stepsize * grad;
    return theta;
};

function optimize() {
    if (run_anim) { return; }		// avoid double-running, e.g., when button is clicked while anim is running 

    sigma = parseFloat(sigmaSlider.value);
    let theta = parseFloat(thetaSlider.value);
    let thetaFirstOrder = thetaSecondOrder = theta;
    let epochs = parseInt(epochsSlider.value);
    let stepsize = parseFloat(stepsizeSlider.value);
    let numSamples = parseInt(numSamplesSlider.value);

    run_anim = true;
    let gt_theta = 0.0;
    Plotly.update('plot', { x: [[thetaFirstOrder]], y: [[stepEdge(thetaFirstOrder)]] }, {}, 3);
    Plotly.update('plot', { visible: true }, {}, 3);
    Plotly.update('plot', { x: [[thetaSecondOrder]], y: [[stepEdge(thetaSecondOrder)]] }, {}, 4);
    Plotly.update('plot', { visible: true }, {}, 4);

    let i = 0;
    const updateTrace = () => {
        if (i < epochs && run_anim) {
            thetaFirstOrder = optimizeFristOrder(sigma, thetaFirstOrder, stepsize, numSamples, antithetic_checkbox.checked, gt_theta);
            thetaSecondOrder = optimizeFristOrder(sigma, thetaSecondOrder, stepsize * 3, numSamples*2, antithetic_checkbox.checked, gt_theta);
            let costFirstOrder = mse(stepEdge(thetaFirstOrder), gt_theta);
            let costSecondOrder = mse(stepEdge(thetaSecondOrder), gt_theta);
            update_trajectory([thetaFirstOrder, stepEdge(thetaFirstOrder), costFirstOrder,
                thetaSecondOrder, stepEdge(thetaSecondOrder), costSecondOrder
            ]);
            text_epochs.value = 'Current Step: ' + (i + 1).toString();
            text_cost.value = 'Current Cost first order (grey): ' + costFirstOrder.toString() + '.0';
            text_cost2.value = 'Current Cost second order (blue): ' + costSecondOrder.toString() + '.0';

            // make timeout so that display is able to react 
            setTimeout(updateTrace, 5);
            i++;
        }
    }

    // call function 
    updateTrace();
}

const main_plot = document.getElementById('plot');
const update_trajectory = (values) => {
    const ID1st = 3;
    const ID2nd = 4;
    const trajFirstOrder = main_plot?.data?.[ID1st];
    const trajSecondOrder = main_plot?.data?.[ID2nd];
    let xData1st = trajFirstOrder.x;
    let yData1st = trajFirstOrder.y;
    let xData2nd = trajSecondOrder.x;
    let yData2nd = trajSecondOrder.y;
    xData1st.push(values[0]);
    yData1st.push(values[1]);
    xData2nd.push(values[3]);
    yData2nd.push(values[4]);
    Plotly.update('plot', { x: [xData1st], y: [yData1st] }, {}, ID1st);
    Plotly.update('plot', { x: [xData2nd], y: [yData2nd] }, {}, ID2nd);
    update_triangle([values[0], values[3]]);
}


defaults = { 'sigma': 1.0, 'nsamples': 10, 'epochs': 600, 'stepsize': 0.1, 'theta': -2.0 };
const FirstOrderColor = 'rgb(234,65,54)';
const SecondOrderColor = 'rgb(104,180,249)';
// Define the data for the Gaussian distribution
let x = [], y_gauss = [], y_gradgauss = [], y_step = [], sigma = 1;
let y_hessgauss = [];
for (let i = -5; i < 5; i += 0.01) {
    x.push(i);
    y_step.push(stepEdge(i));
    y_gauss.push(calcGaussian1D(i, sigma = sigma));
    y_gradgauss.push(calcGradGaussian1DPDF(i, sigma = sigma));
    y_hessgauss.push(calcHessGaussian1DPDF(i, sigma = sigma));
}

// Create the initial plot, declare all the traces 
const gaussianTrace = {
    x: x,
    y: y_gauss,
    name: 'Gaussian',
    type: 'scatter',
    opacity: 0.25
};
const gradGaussianTrace = {
    x: x,
    y: y_gradgauss,
    name: 'Grad. of Gaussian',
    type: 'scatter',
    marker: { color: 'rgb(0, 0, 0)' }
};
const hessGaussianTrace = {
    x: x,
    y: y_hessgauss,
    name: 'Hessian of Gaussian',
    type: 'scatter',
    marker: { color: 'rgb(0, 0, 0)' }
};
const stepTrace = {
    x: x,
    y: y_step,
    name: 'Cost Function',
    type: 'scatter',
    marker: { color: 'orange' }
};
const sampleTrace = {
    x: [],
    y: [],
    name: 'Samples',
    showlegend: false,
    mode: 'markers',
    opacity: 0.5,
    marker: {
        size: 7,
        symbol: 'diamond',
        color: 'black'
    }
};
const sampleTrace_gg = {
    x: [],
    y: [],
    name: 'Samples_gg',
    showlegend: false,
    mode: 'markers',
    opacity: 0.8,
    marker: {
        size: 7,
        symbol: 'diamond',
        color: 'black'
    }
};
const smoothedFn = {
    x: [],
    y: [],
    name: 'Smoothed with gradient samples',
    type: 'scatter',
    opacity: 0.5,
    marker: { color: FirstOrderColor }
};
const smoothedFn2 = {
    x: [],
    y: [],
    name: 'Smoothed with Hessian samples',
    type: 'scatter',
    opacity: 0.5,
    marker: { color: SecondOrderColor }
};
const verticalZero = {
    x: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    y: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    name: '',
    showlegend: false,
    type: 'scatter',
    opacity: 0.75,
    line: { color: 'black', 'width': 0.5 },
};
const pxTrace = {
    x: [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1, 2, 3, 4, 5, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1, 2, 3, 4, 5],
    y: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    name: '',
    showlegend: false,
    type: 'scatter',
    opacity: 0.75,
    marker: { color: 'black', size: 2, line: { color: 'black', width: 2 } }
};
const trajectoryFirstOrder = {
    x: [defaults.theta],
    y: [1.0],
    name: 'Triangle Center First Order',
    type: 'scatter',
    mode: 'markers',
    marker: { color: FirstOrderColor, size: 12, line: { color: FirstOrderColor, width: 1 } }
}; const trajectorySecondOrder = {
    x: [defaults.theta],
    y: [1.0],
    name: 'Triangle Center Second Order',
    type: 'scatter',
    mode: 'markers',
    marker: { color: SecondOrderColor, size: 4, line: { color: SecondOrderColor, width: 1 } }
};
const layout = {
    title: '1D Example: Differentiating Through Plateaus',
    xaxis: { title: 'Theta', 'range': [-5, 5], zeroline: false},
    yaxis: { title: 'Cost', 'range': [-0.1, 1.5] },
    legend: { orientation: 'v', y: 1.0, xanchor: 'right', x: 1.0 },
    margin: { t: 60, b: 60, l: 45, r: 45 },
    /*shapes: [{type: 'rect',
             xref: 'x',
             yref: 'paper',
             x0: -1.5,
             y0: 0.075,
             x1: 1.5,
             y1: 0.15,
             fillcolor: 'royalblue',
             opacity: 0.6,
             layer: 'below',
             line: {width: 0}}],*/
};
const layoutLower1 = {
    title: 'Gradient Samples',
    xaxis: { title: '', 'range': [-5, 5] },
    yaxis: { title: '', 'range': [-0.1, 1.1] },
    legend: { orientation: 'h', y: -0.25, xanchor: 'center', x: 0.5 },
    margin: { t: 45, b: 10, l: 25, r: 10 },
    autosize: true,
};
const layoutLower2 = {
    title: 'Hessian Samples',
    xaxis: { title: '', 'range': [-5, 5] },
    yaxis: { title: '', 'range': [-0.1, 1.1] },
    legend: { orientation: 'h', y: -0.25, xanchor: 'center', x: 0.5 },
    margin: { t: 45, b: 10, l: 25, r: 10 },
    autosize: true,
};
const layoutPxPlot = {
    xaxis: { title: 'Theta', 'range': [-5, 5], zeroline: false, showgrid: false },
    yaxis: {
        title: '', 'range': [-0.2, 0.7], showgrid: false, tickmode: 'array', tickvals: [0],
        showticklabels: false
    },
    legend: { orientation: 'h', y: 0.0, xanchor: 'center', x: 0.5 },
    margin: { t: 2, b: 2, l: 80, r: 80 },
    shapes: [{
        type: 'path',
        path: 'M 0 0 L 1 0.6 L 2 0 Z',
        xref: 'x',
        yref: 'y',
        fillcolor: FirstOrderColor,
        opacity: 1.0,
        line: { width: 1 }
    },
    {
        type: 'path',
        path: 'M 0 0 L 0.1 0.4 L 2 0 Z',
        xref: 'x',
        yref: 'y',
        fillcolor: SecondOrderColor,
        opacity: 1.0,
        line: { width: 1 }
    },
    {
        type: 'rect',
        xref: 'x',
        yref: 'y',
        x0: 0.80,
        y0: 0.1,
        x1: 1.0,
        y1: 0.15,
        fillcolor: 'grey',
        opacity: 0.6,
        line: { width: 1 }
    }],
};
const config = { responsive: true }

Plotly.newPlot('plot', [stepTrace,
    smoothedFn, smoothedFn2,
    trajectoryFirstOrder, trajectorySecondOrder], layout, config);
Plotly.newPlot('plot2', [gaussianTrace,
    gradGaussianTrace,
    sampleTrace,
    sampleTrace_gg, verticalZero], layoutLower1, config);
Plotly.newPlot('plot4', [gaussianTrace,
    hessGaussianTrace,
    sampleTrace,
    sampleTrace_gg, verticalZero], layoutLower2, config);
Plotly.newPlot('plot3', [pxTrace], layoutPxPlot, config);

function reset_textboxes() {
    text_cost.value = 'Current Cost first order (grey): 1.0';
    text_cost2.value = 'Current Cost second order (blue): 1.0';
    text_epochs.value = 'Current Step: 0';
}

function reset(incl_plots = true) {
    stop_anim();
    sigmaSlider.value = defaults.sigma;
    thetaSlider.value = defaults.theta;
    epochsSlider.value = defaults.epochs;
    stepsizeSlider.value = defaults.stepsize;
    numSamplesSlider.value = defaults.nsamples;
    // smoothed_checkbox.checked = false;
    antithetic_checkbox.checked = true;
    reset_textboxes();
    update_triangle();
    
    if (incl_plots) update_plots();
}

var sigmaSlider = document.getElementById('sigma');
var thetaSlider = document.getElementById('theta');
var epochsSlider = document.getElementById('epochs');
var stepsizeSlider = document.getElementById('stepsize');
var numSamplesSlider = document.getElementById('num-samples');
var smoothed_checkbox = document.getElementById('cb_showsmoothed');
var antithetic_checkbox = document.getElementById('cb_antithetic');

var text_cost = document.getElementById('cost_text')
var text_cost2 = document.getElementById('cost_text2')
var text_epochs = document.getElementById('epoch_text')

reset(); 		// set default values to sliders 

var run_anim = false;

thetaSlider.addEventListener('input', function () {
    update_triangle()
    update_plots(resample = false);
    stop_anim();
    reset_textboxes();
});

[epochsSlider, stepsizeSlider].forEach(function (element) {
    element.addEventListener('input', function () {
        update_plots(resample = false);
        stop_anim();
        reset_textboxes();
    });
});

[sigmaSlider, numSamplesSlider, antithetic_checkbox, smoothed_checkbox].forEach(function (element) {
    element.addEventListener('input', function () {
        update_plots(resample = true);
        stop_anim();
        reset_textboxes();
    });
});

function stop_anim() {
    run_anim = false;
}

function theta_to_triPath_tall(th) {
    // expects a single theta parameter, returns a triangle path that is used to update the layout 
    var w = 1.4; 		// tri width 
    var y = 0.0;
    var h = 0.6;
    var tripath = 'M ' + (th - w).toString() + ' ' + y.toString() + ' L ' + th.toString() + ' ' + (y + h).toString() + ' L ' + (th + w).toString() + ' ' + y.toString() + ' Z';
    return tripath;
}
function theta_to_triPath_low(th) {
    // expects a single theta parameter, returns a triangle path that is used to update the layout 
    var w = 1.4; 		// tri width 
    var y = 0.0;
    var h = 0.4;
    var tripath = 'M ' + (th - w).toString() + ' ' + y.toString() + ' L ' + th.toString() + ' ' + (y + h).toString() + ' L ' + (th + w).toString() + ' ' + y.toString() + ' Z';
    return tripath;
}
function update_triangle(theta = null) {
    if (!Array.isArray(theta) || theta.length !== 2) {
        theta = [parseFloat(thetaSlider.value), parseFloat(thetaSlider.value)];
    }
    let thetaFirstOrder, thetaSecondOrder;
    thetaFirstOrder = theta[0];
    thetaSecondOrder = theta[1];
    var tripath1st = theta_to_triPath_tall(thetaFirstOrder);
    var tripath2nd = theta_to_triPath_low(thetaSecondOrder);
    var update = { 'shapes[0].path': tripath1st };
    var update2 = { 'shapes[1].path': tripath2nd };
    Plotly.relayout('plot3', update);
    Plotly.relayout('plot3', update2);
}

function update_plots(resample = true) {

    if (!smoothed_checkbox.checked) { 
        Plotly.update('plot', { visible: false }, {}, 1); 
        Plotly.update('plot', { visible: false }, {}, 2);
    } else { 
        Plotly.update('plot', { visible: true }, {}, 1); 
        Plotly.update('plot', { visible: true }, {}, 2);
    }

    sigma = parseFloat(sigmaSlider.value);
    var theta_init = parseFloat(thetaSlider.value);
    var numSamples = parseInt(numSamplesSlider.value);

    // remove (potential) trajectory 
    Plotly.update('plot', { visible: true }, {}, 3);
    Plotly.update('plot', { x: [[theta_init]], y: [[stepEdge(theta_init)]] }, {}, 3);
    Plotly.update('plot', { visible: true }, {}, 4);
    Plotly.update('plot', { x: [[theta_init]], y: [[stepEdge(theta_init)]] }, {}, 4);

    // update gaussian plots: for gradgaussian, plot pdf, to have same scale easier 
    for (var i = 0; i < x.length; i++) {
        y_gauss[i] = calcGaussian1D(x[i], sigma = sigma);
        y_gradgauss[i] = calcGradGaussian1DPDF(x[i], sigma = sigma);
        y_hessgauss[i] = calcHessGaussian1DPDF(x[i], sigma = sigma);
    }
    Plotly.update('plot2', { y: [y_gauss] }, {}, 0,);				// {} is update for layout, 0 is selector index
    Plotly.update('plot2', { y: [y_gradgauss] }, {}, 1,);		// {} is update for layout, 0 is selector index

    Plotly.update('plot4', { y: [y_gauss] }, {}, 0,);				// {} is update for layout, 0 is selector index
    Plotly.update('plot4', { y: [y_hessgauss] }, {}, 1,);		// {} is update for layout, 0 is selector index


    // update samples: get samples and update the plots 
    var nsamples_real = antithetic_checkbox.checked ? Math.floor(numSamples / 2.0) : numSamples;
    if (resample) {
        const [xsampled_gauss, ysampled_gauss] = getGaussianSamples(nsamples_real, sigma, antithetic_checkbox.checked);
        const [x_gradG, y_gradG, pdf_gradG] = getGradGaussianSamples(nsamples_real, sigma, antithetic_checkbox.checked);
        const [x_hessG, y_hessG, pdf_hessG] = getHessGaussianSamples(nsamples_real, sigma, antithetic_checkbox.checked);
        Plotly.update('plot2', { x: [xsampled_gauss], y: [ysampled_gauss] }, {}, 2);
        Plotly.update('plot2', { x: [x_gradG], y: [pdf_gradG] }, {}, 3);
        Plotly.update('plot4', { x: [xsampled_gauss], y: [ysampled_gauss] }, {}, 2);
        Plotly.update('plot4', { x: [x_hessG], y: [pdf_hessG] }, {}, 3);
    }

    if (!smoothed_checkbox.checked) {
        return;
    } else {
        if (!resample) { return; }

        // go through all x's, for every x make a "smoothed" y-val by sampling N pts from the current
        // x coordinate, and then query and avg their fn val 
        const smoothed = []
        const smoothed2 = []
        for (var i = 0; i < x.length; i += 1) {
            const theta = x[i]		// go from -5 to 5 

            // we concolve with the Gaussian, not the grad.gaussian! 
            //const [xg, yg, pdf_g] = getGradGaussianSamples(nsamples_real, sigma, antithetic_checkbox.checked);
            const [xg, yg] = getGaussianSamples(nsamples_real, sigma, antithetic_checkbox.checked);
            const [xg2, yg2] = getGaussianSamples(nsamples_real, sigma, antithetic_checkbox.checked);
            let fn_avg; fn_avg = 0.0;
            let fn_avg2; fn_avg2 = 0.0;
            for (let j = 0; j < xg.length; j += 1) {
                let theta_perturbed = theta - xg[j];
                let theta_perturbed2 = theta - xg2[j];
                fn_avg += stepEdge(theta_perturbed);
                fn_avg2 += stepEdge(theta_perturbed2);
            }

            smoothed.push(fn_avg / numSamples);
            smoothed2.push(fn_avg2 / numSamples);
        }

        Plotly.update('plot', { x: [x], y: [smoothed] }, {}, 1);
        Plotly.update('plot', { x: [x], y: [smoothed2] }, {}, 2);
    }
}

