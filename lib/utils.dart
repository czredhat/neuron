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

List<double> numberLabelToOutputVector(int num, int max) {
  List<double> outputVector = List.filled(max, 0);
  outputVector[num] = 1;
  return outputVector;
}

int binaryLabelVectorToLabelInt(List<double> inputVector) {
  return inputVector.indexWhere((element) => element >= 0.9);
}
