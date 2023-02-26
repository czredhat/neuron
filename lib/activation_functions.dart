import 'dart:math' as math;

typedef ActivationFunction = double Function(double input);

double identity(double x) {
  return x;
}

double sigmoid(double x) {
  return 1 / (1 + math.pow(math.e, -x));
}

double sigmoidMinus1(double x) {
  double v = x - 1;
  return 1 / (1 + math.pow(math.e, -v));
}

ActivationFunction derActivation(ActivationFunction func) {
  if (func == identity) {
    return derivatedIdentity;
  }
  if (func == sigmoid) {
    return derivatedSigmoid;
  }
  if (func == sigmoidMinus1) {
    return derivatedSigmoid;
  }
  throw Exception('$func doesnt have a derivation');
}


// Derivations ---------------

double derivatedSigmoid(double x) {
  // e^(-x) / (1 + e^(-x))^2
  return math.pow(math.e, -x) / math.pow(1 + math.pow(math.e, -x), 2);
}

double derivatedIdentity(double x) {
  return 1;
}

