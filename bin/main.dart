import 'dart:math' as math;

import '../lib/activation_functions.dart';
import '../lib/loss_functions.dart';
import '../lib/utils.dart';

var random = math.Random(1);



class Layer {

  int inputsCount;

  /// [number of all possible input variants][one concrete input vector]
  List<List<double>> inputs = [];

  /// number of neurons in this layer
  int neurons;

  /// [number of neurons in this layer][number of inputs to this layer]
  List<List<double>> weights = [];


  // derivates with respect to a coresponding weight
  // [number of all possible input variants][number of neurons][number of weights]
  List<List<List<double>>> derivates = [];

  // biases for all neurons in the layer
  List<double> biases = [];

  ActivationFunction activation;

  /// [input index][neuron index] = sum of neuron's i*w
  List<List<double>> iwSum = [];

  /// [number of neurons in this layer][output vector for the given input vector]
  List<List<double>> outputs = [];

  Layer(this.inputsCount, this.neurons, this.activation) {
    // weights initialization

    if (inputsCount <= 0 || neurons < 1) {
      throw Exception('wrong layer parameters ...');
    }

    weights = [];
    for (var ni = 0; ni < neurons; ni++) {
      weights.add(List.generate(inputsCount, (_) => 0 /*random.nextDouble()*/, growable: false));
    }

    // biases initialization
    biases = List.filled(neurons, 0, growable: false);
    for (var ni = 0; ni < neurons; ni++) {
      //biases[ni] = random.nextDouble();
    }
  }

  void compute() {
    // all outputs to 0
    outputs = List.generate(
        inputs.length, (_) => List.filled(neurons, 0, growable: false),
        growable: false);

    // all iwSum to 0
    iwSum = List.generate(
        inputs.length, (_) => List.filled(neurons, 0, growable: false),
        growable: false);

    // calculating output for all inputs variants
    for (int ii = 0; ii < inputs.length; ii++) {
      List<double> input = inputs[ii];

      for (int ni = 0; ni < neurons; ni++) {
        // w * i
        for (int wi = 0; wi < input.length; wi++) {
          iwSum[ii][ni] += weights[ni][wi] * input[wi];
        }

        outputs[ii][ni] = activation(iwSum[ii][ni] + biases[ni]);
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

  void infoDerivates() {
    print('--- derivates:');
    for (int i = 0; i < inputs.length; i++) {
      print('input $i ${inputs[i]} -> ${derivates[i]}');
    }
  }
}

/// Forward propagation of net's inputs to outputs
void solveNet(List<Layer> net) {

  for (int i = 0; i < net.length; i ++) {
    Layer layer = net[i];

    if (i > 0) {
      layer.inputs = net[i -1].outputs;
    }
    layer.compute();
  }
}

double evaluateLoss(List<List<double>> netResults, List<List<double>> wantedResults, LossFunction lossFunction) {

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


void learn(List<Layer> net, List<List<double>> wantedResults, LossFunction lossFunction, [double lr = 0.05]) {

  for (int li = net.length - 1; li >= 0; li --) {

    List<Layer> partialNet = [];
    for (int i = li; i < net.length; i ++) {
      partialNet.add(net[i]);
    }
    Layer layer = partialNet.first;
    Layer lastLayer = net.last;


    layer.derivates = List.generate(layer.inputs.length, (_) {
      return List.generate(layer.neurons, (_) {
        return List.filled(layer.weights[0].length, 0, growable: false);
      }, growable: false);
    }, growable: false);


    for (int ni = 0; ni < layer.neurons; ni ++) {

      for (int wi = 0; wi < layer.weights[ni].length; wi ++) {


        for (int ii = 0; ii < layer.inputs.length; ii ++) {


          for (int woi = 0; woi < layer.weights[ni].length; woi ++) {
            layer.derivates[ii][ni][woi] = 0;
          }

          double inputSum = layer.iwSum[ii][ni];
          layer.derivates[ii][ni][wi] = derActivation(layer.activation)(inputSum) * layer.inputs[ii][wi];

          // vsechny ostatni derivace jine vahy nez wi musim v teto vrstve oznacit jako nula
        }

        // tady musim mit vypocitanou derivaci podle wi pro vsechny vstupy a vystupy site
        double lossDerivation = 0;

        for (int ii = 0; ii < wantedResults.length; ii ++) {
          List<double> wanted = wantedResults[ii];
          List<double> result = lastLayer.outputs[ii];

          for (int oi = 0; oi < result.length; oi++) {
            lossDerivation += derLoss(lossFunction)(wanted[oi], result[oi]) * layer.derivates[ii][oi][wi];
          }
        }

        layer.weights[ni][wi] = layer.weights[ni][wi] - lr * lossDerivation;
        solveNet(partialNet); // po zmene kazde vahy je potreba prepocitat sit

      }

    }

  }

}


void main() {


  // y = 2x + 3y
/*
  List<List<double>> inputs = [];
  List<List<double>> wantedResults = [];

  for (int i = 0; i < 6; i ++) {
    inputs.add([(i * 2).toDouble(), (i * 3).toDouble()]);
    wantedResults.add([(i*2 - i*3.75).toDouble()]);
  }
*/
/*
  List<List<double>> inputs = [];
  List<List<double>> wantedResults = [];
  int k = 10;
  for (int i = 0; i < k; i ++) {
    inputs.add([i.toDouble() * (1/k)] );
    wantedResults.add([(i).toDouble() * (1/k)]);
  }
*/

/*
  // y = 2x
  List<List<double>> inputs = [];
  List<List<double>> wantedResults = [];

  for (int i = 0; i < 3; i ++) {
    inputs.add([i.toDouble()]);
    wantedResults.add([(i).toDouble()]);
  }
*/

/*
  List<List<double>> inputs = [];
  List<List<double>> wantedResults = [];
  int k = 10;
  for (int i = 0; i < k; i ++) {
    inputs.add([i.toDouble() * (1/k), i.toDouble() * (1/k)] );
    wantedResults.add([(i).toDouble() * (1/k), (i).toDouble() * (1/k)]);
  }
*/
/*
  List<List<double>> inputs = [];
  List<List<double>> wantedResults = [];
  int k = 20;
  for (int i = 0; i < k; i ++) {
    inputs.add([i.toDouble() * (1/k)] );
    wantedResults.add([(i).toDouble() * (1/k)]);
  }
*/

  List<List<double>> inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ];
  List<List<double>> wantedResults = [[0, 0, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1] ]; // OR and AND gate


/*
  List<List<double>> inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ];
  List<List<double>> wantedResults = [[0], [0], [0], [1] ]; // OR and AND gate
*/

  // single perceptron net
  List<Layer> net = [
    Layer(inputs[0].length, wantedResults[0].length, identity),
  ];

  net.first.inputs = inputs;

  /*
  net.first.infoWeightsAndBiases();
  return ;
*/

  int start = DateTime.now().millisecondsSinceEpoch;

  for (int step = 0; step < 20; step ++) {
    print('------------------- STEP $step -----------------------');
    solveNet(net);

    for (int i = 0; i < inputs.length; i ++) {
      print('input $i: ${inputs[i]} -> ${wantedResults[i]} vs ${net.last.outputs[i]} ');
    }
    print('Loss: ${evaluateLoss(net.last.outputs, wantedResults, simpleLoss)}');

    if (isItClassifiedWell(wantedResults, net.last.outputs) == true) {
      break;
    }

    learn(net, wantedResults, simpleLoss, 0.05);
  }


  print('last layer info ----');
  net.last.infoWeightsAndBiases();
  //net.last.infoOutputs();
  //net.last.infoDerivates();



  int duration = DateTime.now().millisecondsSinceEpoch - start;
  print('--- Finished in $duration ms');
}
