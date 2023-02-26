import 'dart:math' as math;

import '../lib/activation_functions.dart';
import '../lib/loss_functions.dart';
import '../lib/utils.dart';

var random = math.Random(1);



class Layer {

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
  List<List<double>> iwSum = [];

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

    derivatives = List.generate(
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

void solveDerivatives(List<Layer> net, int neuronIndex, int weightIndex) {

  for (int li = 0; li < net.length; li ++) {
    Layer layer = net[li];

    for (int ni = 0; ni < layer.neurons; ni ++) {
      for (int wi = 0; wi < layer.weights[ni].length; wi ++) {
        for (int ii = 0; ii < layer.inputs.length; ii ++) {

          if (li == 0) {
            // it can be optimized ... the first layer can be computed separately outside this cycle and it doesnt need to iterate all its weights;
            // the first layer of the net has specific derivatives
            if (ni == neuronIndex) {
              layer.derivatives[ii][ni] = derActivation(layer.activation)(layer.iwSum[ii][neuronIndex]) * layer.inputs[ii][weightIndex];
            } else {
              layer.derivatives[ii][ni] = 0;
            }
          } else {
            // TODO
            //layer.derivatives[inputIndex][neuronIndex] = derActivation(layer.activation)(layer.iwSum[inputIndex][neuronIndex]) ;
          }
        }

      }
    }
  }
}


void learn(List<Layer> net, List<List<double>> wantedResults, LossFunction lossFunction, [double lr = 0.05]) {

  // moving backward through the layers
  for (int fli = net.length - 1; fli >= 0; fli --) {
    List<Layer> partialNet = [];
    for (int i = fli; i < net.length; i ++) {
      partialNet.add(net[i]);
    }
    Layer layerToLearn = partialNet.first;
    Layer lastLayer = net.last;


    // I need to compute derivatives here

    for (int ni = 0; ni < layerToLearn.neurons; ni ++) {
      for (int wi = 0; wi < layerToLearn.weights[ni].length; wi ++) {
        print('--- ucim neuron $ni a vahu $wi');

        solveDerivatives(partialNet, ni, wi);

        // tady musim mit vypocitanou derivaci podle wi pro vsechny vstupy a vystupy site
        double lossDerivation = 0;

        for (int ii = 0; ii < wantedResults.length; ii ++) {
          List<double> wanted = wantedResults[ii];
          List<double> result = lastLayer.outputs[ii];

          for (int oi = 0; oi < result.length; oi++) {
            lossDerivation += derLoss(lossFunction)(wanted[oi], result[oi]) * lastLayer.derivatives[ii][oi];
          }
        }

        layerToLearn.weights[ni][wi] = layerToLearn.weights[ni][wi] - lr * lossDerivation;

        layerToLearn.infoWeightsAndBiases();

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



  int duration = DateTime.now().millisecondsSinceEpoch - start;
  print('--- Finished in $duration ms');
}
