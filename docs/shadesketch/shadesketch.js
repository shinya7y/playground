// Reference pages:
// https://github.com/qyzdao/ShadeSketch
// http://hal322.html.xdomain.jp/TensorFlowJS/300_cifar_with_TensorFlowJS.html
// http://www.kogures.com/hitoshi/javascript/canvas/image-local.html

if ( location.href.match('github')) {
  modelDir = './tfjs_models/';
} else {
  modelDir = 'http://localhost:8080/shadesketch/tfjs_models/';  // for debug
}
const normModelPath = modelDir + 'linenorm32/model.json';
const shaderModelPath = {
  '8' : modelDir + 'lineshader8/model.json',
  '16': modelDir + 'lineshader16/model.json',
  '32': modelDir + 'lineshader32/model.json',
};
var currentShaderModelPath = null;
var normModel = null;
var shaderModel = null;

resetBackend();

async function resetBackend() {
  tf.disposeVariables();
  var backendName = document.getElementById('backend_name').value;
  if (backendName !== 'auto') {
    await tf.setBackend(backendName).catch((err) => {
      console.error(err);
      backendName = 'auto';
    });
  }

  if (backendName === 'auto') {
    tf.backend();
    await tf.ready();
  }

  console.log('backend:', tf.getBackend());
  tf.enableProdMode();
}

async function loadModels(shaderBit) {
  if (normModel === null) {
    console.log('Start loading normModel');
    normModel = await tf.loadGraphModel(normModelPath);
    console.log('Finish loading normModel');
  }
  if (currentShaderModelPath !== shaderModelPath[shaderBit]) {
    console.log('Start loading shaderModel');
    currentShaderModelPath = shaderModelPath[shaderBit];
    shaderModel = await tf.loadGraphModel(currentShaderModelPath);
    console.log('Finish loading shaderModel');
  }
}

function cond_to_pos(cond) {
  const cond_pos_rel = {
    '002': [0, 0, -1],
    '110': [0, 1, -1], '210': [1, 1, -1], '310': [1, 0, -1], '410': [1, -1, -1],
    '510': [0, -1, -1], '610': [-1, -1, -1], '710': [-1, 0, -1], '810': [-1, 1, -1],
    '120': [0, 1, 0], '220': [1, 1, 0], '320': [1, 0, 0], '420': [1, -1, 0],
    '520': [0, -1, 0], '620': [-1, -1, 0], '720': [-1, 0, 0], '820': [-1, 1, 0],
    '130': [0, 1, 1], '230': [1, 1, 1], '330': [1, 0, 1], '430': [1, -1, 1],
    '530': [0, -1, 1], '630': [-1, -1, 1], '730': [-1, 0, 1], '830': [-1, 1, 1],
    '001': [0, 0, 1]
  };
  return cond_pos_rel[cond];
}

async function getLightPosition() {
  var lightPositionStr = '810'
  const lpos = document.getElementsByName('lpos');
  for (let i=0; i<lpos.length; i++) {
    if (lpos[i].checked){
      lightPositionStr = lpos[i].value;
      break;
    }
  }
  const lightPosition = cond_to_pos(lightPositionStr);
  console.log('Light Position:', lightPositionStr, lightPosition);
  return lightPosition;
}

function limitImageSize(size) {
  const maxSizeStr = document.getElementById('max_size').value;
  const maxSize = parseInt(maxSizeStr, 10);
  var limitedSize = size;
  if (size > maxSize) {
    limitedSize = maxSize;
  }
  return limitedSize;
}

function loadSampleImage(canvasName) {
  var canvas = document.getElementById(canvasName);
  var context = canvas.getContext('2d');
  var chosenImage = new Image();
  var sampleImages = [
    './sample_images/1.png',
    './sample_images/6.png',
    './sample_images/56736941_p1_line.png',
    './sample_images/5895302_p0_line.png',
    './sample_images/56537426_p0_line.png',
  ];
  chosenImage.src = sampleImages[Math.floor(Math.random() * sampleImages.length)];
  chosenImage.onload = function() {
    canvas.width  = limitImageSize(chosenImage.naturalWidth);
    canvas.height = limitImageSize(chosenImage.naturalHeight);
    context.drawImage(chosenImage, 0, 0, canvas.width, canvas.height);
  }
}

function loadLocalImage(files, canvasName) {
  var canvas = document.getElementById(canvasName);
  var context = canvas.getContext('2d');
  var reader = new FileReader();
  reader.onload = function(event) {
    var chosenImage = new Image();
    chosenImage.onload = function() {
      canvas.width  = limitImageSize(chosenImage.naturalWidth);
      canvas.height = limitImageSize(chosenImage.naturalHeight);
      context.drawImage(chosenImage, 0, 0, canvas.width, canvas.height);
    }
    chosenImage.src = event.target.result;
  }
  reader.readAsDataURL(files[0]);
}

async function executeShading() {
  const lightPosition = await getLightPosition();
  const lineCanvas = document.getElementById('line_canvas');
  const imageW = lineCanvas.width;
  const imageH = lineCanvas.height;
  var shadeCanvas = document.getElementById('shade_canvas');
  shadeCanvas.width  = imageW;
  shadeCanvas.height = imageH;
  var overlayCanvas = document.getElementById('overlay_canvas');
  overlayCanvas.width  = imageW;
  overlayCanvas.height = imageH;

  const rgbLineTensor = tf.browser.fromPixels(lineCanvas).div(tf.scalar(255));
  // RGB2GRAY
  const lineTensor = tf.sum(rgbLineTensor.mul(tf.tensor1d([0.299, 0.587, 0.114])), 2);
  tf.dispose(rgbLineTensor);

  const shaderBit = document.getElementById('shader_bit').value;
  await loadModels(shaderBit);

  // Line norm
  console.log('Start line normalization');
  const normTensor = tf.tidy(() => {
    const normResult = normModel.predict(lineTensor.reshape([1, imageH, imageW, 1]));
    // inverse black-in-white lines to white-in-black
    return tf.sub(tf.scalar(1), normResult);
  });
  console.log('Finish line normalization');

  // Line shade
  console.log('Start shading');
  const startTime = performance.now();
  const shaderTensor = tf.tidy(() => {
    const lightPositionTensor = tf.tensor2d(lightPosition, [1, 3]);
    const shaderPredTensor = shaderModel.predict({
      'input_1' : lightPositionTensor,
      'input_2' : normTensor
    }).squeeze([0, 3]);
    // inverse white-in-black shadow to black-in-white
    // -1 -> 1 (white background), 1 -> 0 (black shadow)
    return shaderPredTensor.mul(tf.scalar(-0.5)).add(tf.scalar(0.5));
  });
  const finishTime = performance.now();
  const elapsedTime = finishTime - startTime;
  console.log('Finish shading\nTime:%f msec', elapsedTime);

  const overlayTensor = tf.add(lineTensor.mul(tf.scalar(0.8)), shaderTensor.mul(tf.scalar(0.2)));
  await tf.browser.toPixels(shaderTensor, shadeCanvas);
  await tf.browser.toPixels(overlayTensor, overlayCanvas);

  tf.dispose(lineTensor);
  tf.dispose(normTensor);
  tf.dispose(shaderTensor);
  tf.dispose(overlayTensor);
}
