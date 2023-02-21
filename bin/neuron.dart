import 'package:neuron/neuron.dart' as neuron;
import 'dart:math' as math;


typedef Activation = double Function(double input);
typedef LossFunction = double Function(double wanted, double result);

double identity(double input) {
  return input;
}

double sigmoid(double input) {
  return 1 / ( 1 + math.pow(math.e, -input));
}

int stepFunction(double input) {
  return input >= 0 ? 1 : 0;
}

int classifyFunction(double input) {
  return input >= 0.5 ? 1 : 0;
}

double simpleLoss(double wanted, double result) {
  return math.pow(wanted - result, 2).toDouble();
}


class Neuron {
  
  List<double> weights;
  double bias;
  Activation activation;

  Neuron(this.weights, this.bias, this.activation );
}

double neuronOutput(List<double> input, List<double> weights, double bias, Activation activation) {

  if (input.length != weights.length) {
  throw Exception('inputs are not the same length as weights');
  }

  double sum = 0;

  for (int k = 0; k < input.length; k ++) {
  sum += input[k] * weights[k];
  }

  //print('-------- $sum');
  return activation(sum) + bias;
}



double loss(List<double> wanted, List<double> results, LossFunction lossFunction) {

  if (wanted.length != results.length) {
    throw Exception('loss function has to have two arrays of the same length');
  }

  double sum = 0;
  for (int i =0; i < wanted.length; i ++) {
    sum += lossFunction(wanted[i], results[i]);
  }

  return sum;
}



void learn(List<List<double>> inputs, Neuron neuron, List<double> wanted, LossFunction lossFunction) {

  double learningRate = 0.05;
  double dx = 0.000000000001;

  List<double> weights = neuron.weights;
  double bias = neuron.bias;

  // updating weights
  for (int wi = 0; wi < weights.length; wi ++) {

    double err = 0;
    {
      List<double> outputs = [];
      for (int i = 0; i < inputs.length; i ++) {
        outputs.add(neuronOutput(inputs[i], weights, bias, neuron.activation));
      }
      err = loss(wanted, outputs, simpleLoss);
    }

    // derivace

    List<double> nWeights = List.from(weights);
    nWeights[wi] = weights[wi] + dx;

    double nErr = 0;
    {
      List<double> outputs = [];
      for (int i = 0; i < inputs.length; i ++) {
        outputs.add(neuronOutput(inputs[i], nWeights, bias, neuron.activation));
      }
      nErr = loss(wanted, outputs, simpleLoss);
    }

    double d = (nErr - err) / dx;

    // updating weight

    double newWeight = weights[wi] - learningRate * d;
    weights[wi] = newWeight;
  }

  neuron.weights = weights;


  // updating bias
  for (int wi = 0; wi < weights.length; wi ++) {

    double err = 0;
    {
      List<double> outputs = [];
      for (int i = 0; i < inputs.length; i ++) {
        outputs.add(neuronOutput(inputs[i], weights, bias, neuron.activation));
      }
      err = loss(wanted, outputs, simpleLoss);
    }

    // derivace

    double nBias = bias + dx;

    double nErr = 0;
    {
      List<double> outputs = [];
      for (int i = 0; i < inputs.length; i ++) {
        outputs.add(neuronOutput(inputs[i], weights, nBias, neuron.activation));
      }
      nErr = loss(wanted, outputs, simpleLoss);
    }

    double d = (nErr - err) / dx;

    // updating bias

    bias = bias - learningRate * d;
  }

  neuron.bias = bias;



}

void main(List<String> arguments) {
  print('Hello world:');

  List<List<double>> inputs = [
    [0,0],
    [0,1],
    [1,0],
    [1,1],
  ];
  List<double> wanted = [0, 1, 1, 1];

  Neuron n1 = Neuron([0, 0], 0, identity);
  n1.weights = [0, 0];
  n1.bias = 0;

  for (int step = 0; step < 50; step ++) {

    print('---- STEP: $step:');


    List<double> outputs = [];

    for (int i = 0; i < inputs.length; i ++) {
      outputs.add(neuronOutput(inputs[i], n1.weights, n1.bias, n1.activation));
    }
    double err = loss(wanted, outputs, simpleLoss);

    print('wanted: ${wanted}');
    print('output: ${outputs}');
    print('loss: ${err}');
    print('weights: ${n1.weights}');
    print('bias: ${n1.bias}');

    bool stop = true;
    for (int i = 0; i < outputs.length; i ++) {
      if (classifyFunction(outputs[i]) != classifyFunction(wanted[i])) {
        stop = false;
        break;
      }
    }
    if (stop == true) {
      print('');
      print('-------- STOP ---------');
      return;
    }



    learn(inputs, n1, wanted, simpleLoss);


  }
}
