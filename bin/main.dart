import 'dart:math' as math;

import 'activations.dart';
import 'loss.dart';

var random = math.Random(1);


int classifyFunction(double input) {
  return input >= 0.5 ? 1 : 0;
}


class Layer {

  int inputsCount;

  // [number of all possible input variants][one concrete input vector]
  List<List<double>> inputs = [];

  // number of neurons in this layer
  int neurons;

  // [number of neurons in this layer][number of inputs to this layer]
  List<List<double>> weights = [];

  // derivates with respect to a coresponding weight
  // [number of all possible input variants][number of neurons][number of weights]
  List<List<List<double>>> derivates = [];

  // biases for all neurons in the layer
  List<double> biases = [];

  ActivationFunction activation;

  // [number of neurons in this layer][output vector for the given input vector]
  List<List<double>> outputs = [];

  Layer(this.inputsCount, this.neurons, this.activation) {
    // weights initialization

    if (inputsCount <= 0 || neurons < 1) {
      throw Exception('wrong layer parameters ...');
    }

    weights = [];
    for (var ni = 0; ni < neurons; ni++) {
      weights.add(List.generate(inputsCount, (_) => random.nextDouble(), growable: false));
    }

    // biases initialization
    biases = List.filled(neurons, 0, growable: false);
    for (var ni = 0; ni < neurons; ni++) {
      //biases[ni] = random.nextDouble();
    }
  }

  double sumForActivation(int inputIndex, int neuronIndex) {

    double sum = 0;
    for (int i = 0; i < weights.length; i ++) {
      sum += inputs[inputIndex][i] * weights[neuronIndex][i];
    }

    return sum;
  }

  void compute() {
    // initialize outputs
    outputs = List.generate(
        inputs.length, (_) => List.filled(neurons, 0, growable: false),
        growable: false);

    // calculating output for all inputs variants
    for (int i = 0; i < inputs.length; i++) {
      List<double> input = inputs[i];

      for (int ni = 0; ni < neurons; ni++) {
        // w * i
        double sum = 0;
        for (int wi = 0; wi < input.length; wi++) {
          sum += weights[ni][wi] * input[wi];
        }
        outputs[i][ni] = activation(sum + biases[ni]);
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


    layer.derivates = List.generate(layer.inputs.length, (_) {
      return List.generate(layer.neurons, (_) {
        return List.filled(layer.weights[0].length, 0, growable: false);
      }, growable: false);
    }, growable: false);

    for (int ni = 0; ni < layer.neurons; ni ++) {
      for (int wi = 0; wi < layer.weights[ni].length; wi ++) {

        for (int ii = 0; ii < layer.inputs.length; ii ++) {

              double inputSum = layer.sumForActivation(ii, ni); // tohle by se dalo optimalizovat ulozenim pro solveNet...

              layer.derivates[ii][ni][wi] = derActivation(layer.activation)(inputSum) * layer.inputs[ii][wi];
          }

        }
    }


    for (int ni = 0; ni < layer.neurons; ni ++) {

      for (int wi = 0; wi < layer.weights[ni].length; wi ++) {

        double lossDerivation = 0;
        Layer lastLayer = net.last;

        for (int ii = 0; ii < wantedResults.length; ii ++) {
          List<double> wanted = wantedResults[ii];
          List<double> result = lastLayer.outputs[ii];

          for (int lni = 0; lni < lastLayer.neurons; lni++) {
            lossDerivation += derLoss(lossFunction)(wanted[lni], result[lni]) * lastLayer.derivates[ii][ni][lni];
          }
        }

        layer.weights[ni][wi] = layer.weights[ni][wi] - lr * lossDerivation;
      }
    }





  }

}

void main() {

  List<List<double>> inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ];

  List<List<double>> wantedResults = [[0], [0], [0], [1] ]; // Identity gate

  // single perceptron net
  List<Layer> net = [
    Layer(2, 1, identity),
  ];

  net.first.inputs = inputs;


  int start = DateTime.now().millisecondsSinceEpoch;

  for (int step = 0; step < 10; step ++) {
    print('------------------- STEP $step -----------------------');
    solveNet(net);

    for (int i = 0; i < inputs.length; i ++) {
      print('input $i: ${inputs[i]} -> ${net.last.outputs[i]}');
    }
    print('Loss: ${evaluateLoss(net.last.outputs, wantedResults, simpleLoss)}');


    bool stop = true;
    for (int i = 0; i < wantedResults.length; i ++) {
      if (classifyFunction(net.last.outputs[i][0]) != wantedResults[i][0] ) {
        stop = false;
        break;
      }
    }

    if (stop == true) {
      break;
    }

    learn(net, wantedResults, simpleLoss, 0.1);
  }

  /*
  print('layer1 info ----');
  net.last.infoWeightsAndBiases();
  net.last.infoOutputs();
  net.last.infoDerivates();
*/


  int duration = DateTime.now().millisecondsSinceEpoch - start;
  print('--- Finished in $duration ms');
}
