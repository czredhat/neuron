
class NetInputOutputCombination {

  List<double> inputVector;

  /// this should be the output for the given input Vector
  List<double> outputVector;

  NetInputOutputCombination(this.inputVector, this.outputVector);
}


class Dataset {

  final List<NetInputOutputCombination> records;

  Dataset(this.records);

  List<List<double>> inputs() {
    return records.map((e) => e.inputVector).toList();
  }

  List<List<double>> outputs() {
    return records.map((e) => e.outputVector).toList();
  }
}