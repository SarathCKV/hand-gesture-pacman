let mobilenet;
let model;
import Webcam from './webcam';
import RPSDataset from './rps-dataset';

const left = document.querySelector('#left');
const up = document.querySelector('#up');
const right = document.querySelector('#right');
const down = document.querySelector('#down');
const doTrain = document.querySelector('#train');
const startPred = document.querySelector('#startPredicting');
const stopPred = document.querySelector('#stopPredicting');

const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var leftSamples = 0,
  upSamples = 0,
  downSamples = 0,
  rightSamples = 0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
  );
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(4);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }),
      tf.layers.dense({ units: 100, activation: 'relu' }),
      tf.layers.dense({ units: 4, activation: 'softmax' }),
    ],
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('Loss: ' + loss);
      },
    },
  });
}

function handleButton(elem) {
  switch (elem) {
    case '0':
      leftSamples++;
      document.getElementById('leftsamples').innerHTML =
        'Left Samples: ' + leftSamples;
      break;
    case '1':
      upSamples++;
      document.getElementById('upsamples').innerHTML =
        'Up Samples: ' + upSamples;
      break;
    case '2':
      rightSamples++;
      document.getElementById('rightsamples').innerHTML =
        'Right Samples: ' + rightSamples;
      break;
    case '3':
      downSamples++;
      document.getElementById('downsamples').innerHTML =
        'Down Samples: ' + downSamples;
      break;
  }
  let label = parseInt(elem);
  const img = webcam.capture();
  dataset.addExample(mobilenet.predict(img), label);
}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictedText = '';
    switch (classId) {
      case 0:
        predictedText = 'Left';
        break;
      case 1:
        predictedText = 'Up';
        break;
      case 2:
        predictedText = 'Right';
        break;
      case 3:
        predictedText = 'Down';
        break;
    }
    // let inputF = document.getElementById('prediction');
    // inputF.value = predictedText;
    // console.log(document.getElementById('prediction').textContent);
    document.getElementById('prediction').innerHTML = predictedText;
    predictedClass.dispose();
    await tf.nextFrame();
  }
}

function doTraining() {
  train();
}

function startPredicting() {
  isPredicting = true;
  predict();
}

function stopPredicting() {
  isPredicting = false;
  predict();
}

async function init() {
  await webcam.setup();
  mobilenet = await loadMobilenet();
  tf.tidy(() => mobilenet.predict(webcam.capture()));
}

init();

left.addEventListener('click', () => handleButton('0'));
up.addEventListener('click', () => handleButton('1'));
right.addEventListener('click', () => handleButton('2'));
down.addEventListener('click', () => handleButton('3'));
doTrain.addEventListener('click', doTraining);
startPred.addEventListener('click', startPredicting);
stopPred.addEventListener('click', stopPredicting);
