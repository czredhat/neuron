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