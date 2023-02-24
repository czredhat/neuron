import 'dart:math' as math;

typedef ActivationFunction = double Function(double input);

double identity(double input) {
  return input;
}

double sigmoid(double input) {
  return 1 / (1 + math.pow(math.e, -input));
}

ActivationFunction derActivation(ActivationFunction func) {
  if (func == identity) {
    return derivatedIdentity;
  }
  throw Exception('$func doesnt have a derivation');
}


// Derivations

double derivatedIdentity(double input) {
  return 1;
}

