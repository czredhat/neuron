import 'dart:convert';
import 'dart:io';

import 'package:archive/archive_io.dart';
import 'package:csv/csv.dart';
import 'package:neuron/dataset.dart';
import 'package:neuron/utils.dart';

import 'dart:math';

enum MnistDatesetType {
  mnistTrain,
  mnistTest,
}

Future<Dataset> loadMnist(MnistDatesetType mnistSet) async {

  String path = 'MNIST/mnist_train.csv';
  if (mnistSet == MnistDatesetType.mnistTrain) {
    path = 'MNIST/mnist_train.csv';
  } else if (mnistSet == MnistDatesetType.mnistTest) {
    path = 'MNIST/mnist_test.csv';
  }

  File mnistFile = File('datasets/unarchived/$path');

  if (mnistFile.existsSync() == false) {
    final inputStream = InputFileStream('datasets/MNIST.zip');
    final archive = ZipDecoder().decodeBuffer(inputStream);

    for (var file in archive.files) {
      if (file.isFile) {
        if (file.name == path) {
          print('--- MNIST train CSV found.');

          if (File('datasets/unarchived/${file.name}').existsSync() == false) {
            print('----- uncompressing ...');
            final outputStream = OutputFileStream('datasets/unarchived/${file.name}');
            file.writeContent(outputStream);
            // Make sure to close the output stream so the File is closed.
            outputStream.close();
          } else {
            print('----- available!');
          }
          continue;
        }
      }
    }
  }

  final input = File('datasets/unarchived/$path').openRead();
  List rows = await input
      .transform(utf8.decoder)
      .transform(CsvToListConverter(
        shouldParseNumbers: true,
        allowInvalid: false,
      ))
      .toList();

  Dataset dataset = Dataset([]);

  for (int r = 1; r < rows.length; r++) {
    var row = rows[r];

    List<double> outputVector = numberLabelToOutputVector(row[0], 10);
    List<double> inputVector = List.filled(row.length - 1, 0);

    for (int c = 0; c < row.length - 1; c++) {
      inputVector[c] = (row[c + 1] as int).toDouble() / 255;
    }

    dataset.records.add(NetInputOutputCombination(inputVector, outputVector));
  }

  print('--- Dataset $path loaded. ${dataset.records.length} records.');
  return dataset;
}

void printDatasetHistogram(Dataset dataset, [int offset = 0, int? length]) {
  if (length == null) {
    length = dataset.records.length;
  }

  List<int> labelHistogram = List.filled(10, 0);
  for (int i = offset; i < offset + length; i ++) {
    labelHistogram[binaryLabelVectorToLabelInt(dataset.records[i].outputVector)] ++;
  }

  print('--- Histogram for $length samples:');
  for (int i = 0; i < labelHistogram.length; i ++) {
    print('$i -> ${labelHistogram[i]}');
  }
}

Dataset getRandonSampleWithUniformHistogram(Dataset dataset, int samples, Random random) {

  List<NetInputOutputCombination> records = [];
  List<List<NetInputOutputCombination>> labeledRecords = [];

  int start = random.nextInt(dataset.records.length);
  int perCategory = samples ~/ 10 + 1;

  for (int i = 0; i < 10; i ++) {

    labeledRecords.add([]);

    int pos = start;
    while (labeledRecords[i].length < perCategory) {

      NetInputOutputCombination record = dataset.records[pos % dataset.records.length];
      if (binaryLabelVectorToLabelInt(record.outputVector) == i) {
        labeledRecords[i].add(record);
      }
      pos ++;
    }
  }

  int pos = random.nextInt(10);
  List<int> increments = List.filled(10, 0);
  while (records.length < samples) {
    int label = pos % 10;
    if (increments[label] < labeledRecords[label].length) {
      records.add(labeledRecords[label][increments[label]]);
      increments[label] ++;
    }

    pos ++;
  }

  return Dataset(records);
}
