let recognizer;
var start=false;
//INFERENCE

function predictWord() {
 // Array of words that the recognizer is trained to recognize.
 const words = recognizer.wordLabels();
 recognizer.listen(({scores}) => {
   // Turn scores into a list of (score,word) pairs.
   scores = Array.from(scores).map((s, i) => ({score: s, word: words[i]}));
   // Find the most probable word.
   scores.sort((s1, s2) => s2.score - s1.score);
 }, {probabilityThreshold: 0.7});
}

// DATA COLLECTION
const NUM_FRAMES = 3;
let examples = [];
let left=0;
let right=0;
let noise=0;




function collect(label) {
 if (label == null) {
   return recognizer.stopListening();
 }
 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   examples.push({vals, label});

     if (label == 0){
       left++;
       document.querySelector('#leftd').textContent =
           `${Math.floor(left/40)} seconds of audio collected`;
        }

     if (label == 1){
       right++;
       document.querySelector('#rightd').textContent =
           `${Math.floor(right/40)} seconds of audio collected`;
        }
     if (label == 2){
       noise++;
       document.querySelector('#noised').textContent =
           `${Math.floor(noise/40)} seconds of audio collected`;
        }

   document.querySelector('#console').textContent =
       `${Math.floor(examples.length/40)} total seconds of audio collected`;
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}

function normalize(x) {
 const mean = -100;
 const std = 10;
 return x.map(x => (x - mean) / std);
}


//TRAIN
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;

async function train() {
 if(examples.length<16){
 document.querySelector('#console').textContent =
           `Please input more data...`;
    return;
 }
 toggleButtons(false);
 const ys = tf.oneHot(examples.map(e => e.label), 3);
 const xsShape = [examples.length, ...INPUT_SHAPE];
 const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

 await model.fit(xs, ys, {
   batchSize: 16,
   epochs: 5,
   callbacks: {
     onEpochEnd: (epoch, logs) => {
       document.querySelector('#console').textContent =
           `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
     }
   }
 });
 tf.dispose([xs, ys]);
 toggleButtons(true);
}

function buildModel() {
 model = tf.sequential();
 model.add(tf.layers.depthwiseConv2d({
   depthMultiplier: 8,
   kernelSize: [NUM_FRAMES, 3],
   activation: 'relu',
   inputShape: INPUT_SHAPE
 }));
 model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
 model.add(tf.layers.flatten());
 model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
 const optimizer = tf.train.adam(0.01);
 model.compile({
   optimizer,
   loss: 'categoricalCrossentropy',
   metrics: ['accuracy']
 });
}

function toggleButtons(enable) {
 document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

function flatten(tensors) {
 const size = tensors[0].length;
 const result = new Float32Array(tensors.length * size);
 tensors.forEach((arr, i) => result.set(arr, i * size));
 return result;
}

//Real-Time Prediction


right = false;
left = false;


async function moveSlider(labelTensor) {

 const label = (await labelTensor.data())[0];
 if (label == 2) {
    right = false;
    left = false;
   return;
 }
 if (label == 1) {
    right=true;
 }
 if (label==0){
    left=true;
 }
 let delta = 0.2;
 const prevValue = +document.getElementById('output').value;
 document.getElementById('output').value =
     prevValue + (label === 0 ? -delta : delta);
}

function listen() {
    var demo = document.querySelector('#demo');
    demo.style.visibility='hidden';
    setInterval(draw1,10);
 if (recognizer.isListening()) {
   recognizer.stopListening();
   toggleButtons(true);
   document.getElementById('listen').textContent = 'Start';
   return;
 }
 toggleButtons(false);
 document.getElementById('listen').textContent = 'Stop';
 document.getElementById('listen').disabled = false;

 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
   const probs = model.predict(input);
   const predLabel = probs.argMax(1);
   await moveSlider(predLabel);
   tf.dispose([input, probs, predLabel]);
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}

//MAIN

async function app() {
 recognizer = speechCommands.create('BROWSER_FFT');
 await recognizer.ensureModelLoaded();
 //predictWord();
  buildModel();
}

app();



//game
var context;
var dx= 2;
var dy= 1;
var y=100;
var x=150;
function draw1(){

    //init
    context= myCanvas.getContext('2d');
    context.clearRect(0,0,300,300);
    context.beginPath();
    context.fillStyle="green"
    context.arc(x,y,10,0,Math.PI*2,true);
    context.closePath();
    context.fill();

    //movement restriction
    if( x<10 || x>290){
        dx=-dx;
    }
    if( y<20 || y>150){
        dy=-dy;}


    //user movement
    if(right===true){
    x+=dx;
    }
    if(left===true){
    x-=dx;
    }

    //gravity
    y+=dy;
    }




