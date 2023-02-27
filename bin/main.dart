import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'package:archive/archive_io.dart';
import 'package:csv/csv.dart';
import 'package:neuron/dataset.dart';
import 'package:neuron/mnist_utils.dart';

import '../lib/activation_functions.dart';
import '../lib/loss_functions.dart';
import '../lib/utils.dart';

var random = math.Random(1);

class Layer {

  // number of inputs to the layer
  int inputsCount;

  /// [input's index][one concrete input vector] = one concrete example of an input
  List<List<double>> inputs = [];

  /// number of neurons in this layer
  int neurons;

  /// [neuron's index][weight's index]
  List<List<double>> weights = [];

  /// [neuron's index] bias for the specified neuron
  List<double> biases = [];

  ActivationFunction activation;

  /// [input's index][neuron's index] = sum of given neuron's i*w for specified input
  List<List<double>> iwbSum = [];

  /// [input's index][neuron's index] = given neuron output value for specified input
  List<List<double>> outputs = [];

  /// [input's index][neuron's index] = given neuron derivative output for specified input
  List<List<double>> derivatives = [];


  Layer(this.inputsCount, this.neurons, this.activation) {
    // weights initialization
    if (inputsCount <= 0 || neurons < 1) {
      throw Exception('wrong layer parameters ...');
    }

    weights = [];
    for (var ni = 0; ni < neurons; ni++) {
      weights.add(List.generate(inputsCount, (_) => random.nextDouble(), growable: false));
      //weights.add(List.filled(inputsCount, 0, growable: false));
    }

    // biases initialization
    biases = List.filled(neurons, 0, growable: false);
    for (var ni = 0; ni < neurons; ni++) {
      biases[ni] = random.nextDouble();
    }
  }

  void compute() {
    /// computes tha layer outputs according to its input
    outputs = List.generate(
        inputs.length, (_) => List.filled(neurons, 0, growable: false),
        growable: false);

    // all iwSum to 0
    iwbSum = List.generate(
        inputs.length, (_) => List.filled(neurons, 0, growable: false),
        growable: false);

    derivatives = List.generate(
        inputs.length, (_) => List.filled(neurons, 0, growable: false),
        growable: false);

    // calculating output for all inputs variants
    for (int ii = 0; ii < inputs.length; ii++) {
      List<double> input = inputs[ii];

      for (int ni = 0; ni < neurons; ni++) {
        // w * i
        for (int wi = 0; wi < input.length; wi++) {
          iwbSum[ii][ni] += weights[ni][wi] * input[wi];
        }

        iwbSum[ii][ni] += biases[ni];
        outputs[ii][ni] = activation(iwbSum[ii][ni]);
      }
    }
  }

  void infoWeightsAndBiases() {
    print('--- weights and biases:');
    for (int ni = 0; ni < neurons; ni++) {
      String text = 'n$ni: w: ${weights[ni]} ';
      text += 'b: ${biases[ni]} ';
      print(text);
    }
  }

  void infoOutputs() {
    print('--- outputs:');
    for (int i = 0; i < inputs.length; i++) {
      print('input $i ${inputs[i]} -> ${outputs[i]}');
    }
  }

}

/// Forward propagation of net's inputs to outputs
void solveNet(List<Layer> net) {

  for (int li = 0; li < net.length; li ++) {
    Layer layer = net[li];

    if (li > 0) {
      layer.inputs = net[li -1].outputs;
    }
    layer.compute();
  }
}

double evaluateLoss(List<List<double>> netResults, List<List<double>> wantedResults, LossFunction lossFunction) {
  /// returns the loss error of the network
  if (netResults.length != wantedResults.length || netResults[0].length != wantedResults[0].length) {
    throw Exception('bad size of parameters!');
  }

  double sum = 0;
  for (int i = 0; i < netResults.length; i ++) {
    for (int r = 0; r < netResults[0].length; r ++) {
      sum += lossFunction(wantedResults[i][r], netResults[i][r]);
    }
  }
  return sum;
}

void solveDerivativesInDeeperLayers(List<Layer> net) {
  /// solves derivatives in deeper layers of the net
  for (int li = 1; li < net.length; li ++) {
    Layer layer = net[li];
    Layer prevLayer = net[li - 1];

    for (int ni = 0; ni < layer.neurons; ni ++) {
      for (int ii = 0; ii < layer.inputs.length; ii ++) {

        double sum = 0;
        for (int wi = 0; wi < layer.weights[ni].length; wi ++) {
          sum += prevLayer.derivatives[ii][wi] * layer.weights[ni][wi];
        }
        layer.derivatives[ii][ni] = derActivation(layer.activation)(layer.iwbSum[ii][ni]) * sum;
      }
    }
  }
}

void solveDerivativesRespectingWeight(List<Layer> net, int neuronIndex, int weightIndex) {
  /// solves derivatives for the first layer in the net with respect to given weight
  Layer layer = net[0];
  for (int ni = 0; ni < layer.neurons; ni ++) {
    for (int ii = 0; ii < layer.inputs.length; ii ++) {
      if (ni == neuronIndex) {
        layer.derivatives[ii][ni] = derActivation(layer.activation)(layer.iwbSum[ii][ni]) * layer.inputs[ii][weightIndex];
      } else {
        layer.derivatives[ii][ni] = 0;
      }
    }
  }
  solveDerivativesInDeeperLayers(net);
}

void solveDerivativesRespectingBias(List<Layer> net, int neuronIndex) {
  /// solves derivatives for the first layer in the net with respect to given bias
  Layer layer = net[0];
  for (int ni = 0; ni < layer.neurons; ni ++) {
    for (int ii = 0; ii < layer.inputs.length; ii ++) {
      if (ni == neuronIndex) {
        layer.derivatives[ii][ni] = derActivation(layer.activation)(layer.iwbSum[ii][ni]);
      } else {
        layer.derivatives[ii][ni] = 0;
      }
    }
  }
  solveDerivativesInDeeperLayers(net);
}

double gradientOfLossFunction(List<Layer> net, List<List<double>> wantedResults, LossFunction lossFunction) {
  /// returns the gradient of the loss function
  Layer lastLayer = net.last;
  double lossDerivation = 0;

  for (int ii = 0; ii < wantedResults.length; ii ++) {
    List<double> wanted = wantedResults[ii];
    List<double> result = lastLayer.outputs[ii];

    for (int oi = 0; oi < result.length; oi++) {
      lossDerivation += derLoss(lossFunction)(wanted[oi], result[oi]) * lastLayer.derivatives[ii][oi];
    }
  }
  return lossDerivation;
}

void learn(List<Layer> net, List<List<double>> wantedResults, LossFunction lossFunction, [double lr = 0.05]) {

  // moving backward through the layers
  for (int fli = net.length - 1; fli >= 0; fli --) {
    List<Layer> partialNet = [];
    for (int i = fli; i < net.length; i ++) {
      partialNet.add(net[i]);
    }
    Layer layerToLearn = partialNet.first;

    for (int ni = 0; ni < layerToLearn.neurons; ni ++) {
      for (int wi = 0; wi < layerToLearn.weights[ni].length; wi ++) {

        // gradient of loss function for each weight in the network
        solveDerivativesRespectingWeight(partialNet, ni, wi);
        layerToLearn.weights[ni][wi] = layerToLearn.weights[ni][wi] - lr * gradientOfLossFunction(partialNet, wantedResults, lossFunction);
        solveNet(partialNet);
      }

      // gradient of loss function for each bias in the network
      solveDerivativesRespectingBias(partialNet, ni);
      layerToLearn.biases[ni] = layerToLearn.biases[ni] - lr * gradientOfLossFunction(partialNet, wantedResults, lossFunction);;
      solveNet(partialNet);
    }
  }

}

void main() async {

  print('Hello ------------------');
  Dataset mnistTrain = await loadMnist(MnistDatesetType.mnistTrain);
  Dataset mnistTest = await loadMnist(MnistDatesetType.mnistTest);

  Dataset randomDataset = getRandonSampleWithUniformHistogram(mnistTrain, 30, random);
  printDatasetHistogram(randomDataset);


  return;

  List<List<double>> inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ];
  List<List<double>> wantedResults = [[0], [1], [1], [0] ]; // XOR


  print('inputs: $inputs');
  print('outputs: $wantedResults');

  // single perceptron net
  List<Layer> net = [
    Layer(2, 2, sigmoid),
    Layer(2, 1, identity),
  ];

  net.first.inputs = inputs;


  int start = DateTime.now().millisecondsSinceEpoch;

  for (int step = 0; step < 200; step ++) {
    print('------------------- STEP $step -----------------------');
    solveNet(net);

    /*
    for (int i = 0; i < inputs.length; i ++) {
      print('input $i: ${inputs[i]} -> ${wantedResults[i]} vs ${net.last.outputs[i]} ');
    }
    */

    print('Loss: ${evaluateLoss(net.last.outputs, wantedResults, simpleLoss)}');

    if (isItClassifiedWell(wantedResults, net.last.outputs) == true) {
      break;
    }

    learn(net, wantedResults, simpleLoss, 0.3);
  }

  for (int i = 0; i < inputs.length; i ++) {
    print('input $i: ${inputs[i]} -> ${wantedResults[i]} vs ${net.last.outputs[i]} ');
  }

  //net.last.infoWeightsAndBiases();

  int duration = DateTime.now().millisecondsSinceEpoch - start;
  print('--- Finished in $duration ms');
}
