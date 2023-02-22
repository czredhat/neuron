import 'dart:math' as math;


var random = math.Random(1);

double uniformRandom() {
  return (random.nextDouble() - 0.5) * 2;
}

typedef ActivationFunction = double Function(double input);
typedef LossFunction = double Function(double wanted, double result);

double identity(double input) {
  return input;
}

double sigmoid(double input) {
  return 1 / (1 + math.pow(math.e, -input));
}

double relu(double input) {
  return input > 0? input : 0;
}

double stepFunction(double input) {
  return input >= 0 ? 1 : 0;
}

int classifyFunction(double input) {
  return input >= 0.5 ? 1 : 0;
}

double simpleLoss(double wanted, double result) {
  return math.pow(wanted - result, 2).toDouble();
}

class Layer {

  int inputsCount;

  // [number of all possible input variants][one concrete input vector]
  List<List<double>> inputs = [];

  // number of neurons in this layer
  int neurons;

  // [number of neurons in this layer][number of inputs to this layer]
  List<List<double>> weights = [];

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
      weights.add(List.filled(inputsCount, 0, growable: false));
      for (var wi = 0; wi < inputsCount; wi++) {
        weights[ni][wi] = random.nextDouble();
      }
    }

    // biases initialization
    biases = List.filled(neurons, 0, growable: false);
    for (var ni = 0; ni < neurons; ni++) {
      biases[ni] = random.nextDouble();
    }
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


void learn(List<Layer> net, List<List<double>> wantedResults, LossFunction lossFunction, [double lr = 0.05, double dx = 0.000000000001,]) {

  for (int li = net.length - 1; li >= 0; li --) {
    Layer layer = net[li];

    List<Layer> partialNet = [];
    for (int i = li; i < net.length; i ++) {
      partialNet.add(net[i]);
    }

    // adjusting weights ...
    for (int ni = 0; ni < layer.neurons; ni ++) {
      for (int wi = 0; wi < layer.weights[ni].length; wi ++) {
        double originalWeight = layer.weights[ni][wi];

        // I need to compute the gradient of loss function with respect to the weight
        // this is a really naive method

        solveNet(partialNet);
        double y1Loss = evaluateLoss(partialNet[partialNet.length - 1].outputs, wantedResults, lossFunction);

        layer.weights[ni][wi] = layer.weights[ni][wi] + dx;
        solveNet(partialNet);
        double y2Loss = evaluateLoss(partialNet[partialNet.length - 1].outputs, wantedResults, lossFunction);

        double gradient = (y2Loss - y1Loss) / dx;
        layer.weights[ni][wi] = originalWeight - lr * gradient;
      }
    }

    // adjusting biases ...
    for (int bi = 0; bi < layer.biases.length; bi ++) {
      double originalBias = layer.biases[bi];

      // I need to compute the gradient of loss function with respect to the bias
      // this is a really naive method

      solveNet(partialNet);
      double y1Loss = evaluateLoss(partialNet[partialNet.length - 1].outputs, wantedResults, lossFunction);

      layer.biases[bi] = layer.biases[bi] + dx;
      solveNet(partialNet);
      double y2Loss = evaluateLoss(partialNet[partialNet.length - 1].outputs, wantedResults, lossFunction);

      double gradient = (y2Loss - y1Loss) / dx;
      layer.biases[bi] = originalBias - lr * gradient;
    }
    
  }

}

void main() {
  print('HELLO --------------');

  List<List<double>> inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];

  List<List<double>> wantedResults = [[0], [1], [1], [0]]; // XOR gate

//
  List<Layer> net = [
    Layer(2, 2, sigmoid),
    Layer(2, 1, identity),
  ];

  net.first.inputs = inputs;


  int start = DateTime.now().millisecondsSinceEpoch;

  for (int step = 0; step < 1000; step ++) {
    print('------- STEP $step');
    solveNet(net);
    //print('layer2 info ----');
    //net.last.infoWeightsAndBiases();
    //net.last.infoOutputs();
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

    learn(net, wantedResults, simpleLoss, 0.3, 0.0000000001);
  }


  int duration = DateTime.now().millisecondsSinceEpoch - start;
  print('--- Finished in $duration ms');
}
