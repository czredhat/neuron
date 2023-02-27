int classifyFunction(double input) {
  return input >= 0.5 ? 1 : 0;
}

bool isItClassifiedWell(List<List<double>> wantedResults, List<List<double>> results) {
  bool stop = true;
  for (int ii = 0; ii < wantedResults.length; ii ++) {
    for (int ri = 0; ri < results[ii].length; ri ++) {
      if (classifyFunction(results[ii][ri]) != wantedResults[ii][ri]) {
        stop = false;
        break;
      }
    }
  }
  return stop;
}

List<double> vectorToBinaryClassVector(List<double> vector) {
  double max = 0;
  int index = 0;
  List<double> output = List.filled(vector.length, 0);

  for (int i = 0; i < vector.length; i ++) {
    if (vector[i] > max) {
      max = vector[i];
      index = i;
    }
  }

  output[index] = 1;
  return output;
}

bool areVectorEqual(List<double> v1, List<double> v2) {
  if (v1.length != v2.length) {
    return false;
  }

  bool equals = true;
  for (int i = 0; i < v1.length; i ++) {
    if (v1[i] != v2[i]) {
      equals = false;
      break;
    }
  }
  return equals;
}


List<double> numberLabelToOutputVector(int num, int max) {
  List<double> outputVector = List.filled(max, 0);
  outputVector[num] = 1;
  return outputVector;
}

int binaryLabelVectorToLabelInt(List<double> inputVector) {
  return inputVector.indexWhere((element) => element >= 0.9);
}
