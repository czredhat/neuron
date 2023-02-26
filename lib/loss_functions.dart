import 'dart:math' as math;

typedef LossFunction = double Function(double wanted, double result);

double simpleLoss(double wanted, double result) {
  return math.pow(result - wanted, 2).toDouble();
}

double derSimpleLoss(double wanted, double result) {
  // (o - y)^2
  // 2*(o - y) * der(o - y)

  return 2*(result - wanted);
}


LossFunction derLoss(LossFunction func) {
  if (func == simpleLoss) {
    return derSimpleLoss;
  }
  throw Exception('$func doesnt have a derivation');
}