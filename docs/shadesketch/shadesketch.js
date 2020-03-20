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
var shouldLoadedModels = true;
var normModel, shaderModel;

async function loadModels(shaderBit) {
  console.log("Start loading model");
  normModel = await tf.loadGraphModel(normModelPath);
  shaderModel = await tf.loadGraphModel(shaderModelPath[shaderBit]);
  console.log("Finish loading model");
  shouldLoadedModels = false;
}

function cond_to_pos(cond) {
  const cond_pos_rel = {
    '002': [0, 0, -1],
    '110': [0, 1, -1], '210': [1, 1, -1], '310': [1, 0, -1], '410': [1, -1, -1], '510': [0, -1, -1],
    '610': [-1, -1, -1], '710': [-1, 0, -1], '810': [-1, 1, -1],
    '120': [0, 1, 0], '220': [1, 1, 0], '320': [1, 0, 0], '420': [1, -1, 0], '520': [0, -1, 0], '620': [-1, -1, 0],
    '720': [-1, 0, 0], '820': [-1, 1, 0],
    '130': [0, 1, 1], '230': [1, 1, 1], '330': [1, 0, 1], '430': [1, -1, 1], '530': [0, -1, 1], '630': [-1, -1, 1],
    '730': [-1, 0, 1], '830': [-1, 1, 1],
    '001': [0, 0, 1]
  };
  return cond_pos_rel[cond];
}

function limitImageSize(size) {
  const maxSizeStr = document.getElementById("max_size").value;
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
    "./sample_images/1.png",
    "./sample_images/6.png",
    "./sample_images/56736941_p1_line.png",
    "./sample_images/5895302_p0_line.png",
    "./sample_images/56537426_p0_line.png",
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
  var lightPositionStr = "810"
  const lpos = document.getElementsByName("lpos");
  for(let i=0; i<lpos.length; i++) {
    if(lpos[i].checked){
      lightPositionStr = lpos[i].value;
      break;
    }
  }
  const lightPosition = cond_to_pos(lightPositionStr);
  console.log("Light Position:", lightPositionStr, lightPosition);

  const lineCanvas = document.getElementById('line_canvas');
  const lineContext = lineCanvas.getContext('2d');
  const imageW = lineCanvas.width;
  const imageH = lineCanvas.height;
  const lineImageData = lineContext.getImageData(0, 0, imageW, imageH);
  var shadeCanvas = document.getElementById('shade_canvas');
  var shadeContext = shadeCanvas.getContext('2d');
  shadeCanvas.width  = imageW;
  shadeCanvas.height = imageH;
  var shadeImageData = shadeContext.createImageData(imageW, imageH)
  var overlayCanvas = document.getElementById('overlay_canvas');
  var overlayContext = overlayCanvas.getContext('2d');
  overlayCanvas.width  = imageW;
  overlayCanvas.height = imageH;
  var overlayImageData = overlayContext.createImageData(imageW, imageH)

  var grayLineData = new Float32Array(lineImageData.data.length / 4);
  for(let i=0; i<lineImageData.data.length; i+=4) {
    grayLineData[i / 4] = (
      lineImageData.data[i + 0] * 0.299 +
      lineImageData.data[i + 1] * 0.587 +
      lineImageData.data[i + 2] * 0.114) / 255.0;
  }

  const shaderBit = document.getElementById("shader_bit").value;
  if (shouldLoadedModels) {
    await loadModels(shaderBit);
  }

  // Line norm
  console.log("Start line normalization");
  const normTensor = tf.tidy(() => {
    const lineTensor = tf.tensor4d(grayLineData, [1, imageH, imageW, 1]);
    const normResult = normModel.predict(lineTensor);
    // inverse black-in-white lines to white-in-black
    return tf.sub(tf.scalar(1), normResult);
  });
  console.log("Finish line normalization");

  // Line shade
  console.log("Start shading");
  const shaderTensor = tf.tidy(() => {
    const lightPositionTensor = tf.tensor2d(lightPosition, [1, 3]);
    return shaderModel.predict({
      'input_1' : lightPositionTensor,
      'input_2' : normTensor
    });
  });
  const shaderResult = shaderTensor.dataSync();
  console.log("Finish shading");
  tf.dispose(normTensor);
  tf.dispose(shaderTensor);

  // post-process
  for(let y=0; y<imageH; y++) {
    for(let x=0; x<imageW; x++) {
      let i = y * imageW + x;
      // inverse white-in-black shadow to black-in-white
      shadeGrayColor = (1 - (shaderResult[i] + 1) / 2) * 255;
      shadeImageData.data[i*4 + 0] = shadeGrayColor;
      shadeImageData.data[i*4 + 1] = shadeGrayColor;
      shadeImageData.data[i*4 + 2] = shadeGrayColor;
      shadeImageData.data[i*4 + 3] = 255;
      overlayImageData.data[i*4 + 0] = lineImageData.data[i*4 + 0] * 0.8 + shadeGrayColor * 0.2;
      overlayImageData.data[i*4 + 1] = lineImageData.data[i*4 + 1] * 0.8 + shadeGrayColor * 0.2;
      overlayImageData.data[i*4 + 2] = lineImageData.data[i*4 + 2] * 0.8 + shadeGrayColor * 0.2;
      overlayImageData.data[i*4 + 3] = 255;
    }
  }

  shadeContext.putImageData(shadeImageData, 0, 0);
  overlayContext.putImageData(overlayImageData, 0, 0);
}

