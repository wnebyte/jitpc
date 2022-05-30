import argparse
import pandas as pd
from decisiontree import DecisionTree
from knn import KNN
from naivebayes import NaiveBayes
from neuralnetwork import NeuralNetwork
from svm import SVM

"""
Classify samples using pre-existing model

positional arguments:
  {DT,KNN,NB,NN,SVM}   model type
  CLF                  path to serialized clf (.pkl)
  SCL                  path to serialized scaler (.pkl)
  DATA                 path to samples (.csv)

optional arguments:
  -h, --help           show this help message and exit
  -sep SEP             separator char for parsing csv data
"""

def main():
    parser = argparse.ArgumentParser(description="Classify samples with pre-existing model")
    # positional arguments
    parser.add_argument('type', type=str,
                        choices=['DT', 'KNN', 'NB', 'NN', 'SVM'], help='model type')
    parser.add_argument('clf', metavar='CLF', type=str, help='path to serialized clf (.pkl)')
    parser.add_argument('scaler', metavar='SCL', type=str, help='path to serialized scaler (.pkl)')
    parser.add_argument('data', metavar='DATA', type=str, help='path to samples (.csv)')
    # optional arguments
    parser.add_argument('-sep', type=str, dest='sep', default=',', help='separator char for parsing csv data')
    args = parser.parse_args()
    run(args)

def run(args):
    type = args.type
    model = None

    if type == 'DT':
        model = DecisionTree(args.clf, args.scaler)
    elif type == 'KNNB':
        model = KNN(args.clf, args.scaler)
    elif type == 'NB':
        model = NaiveBayes(args.clf, args.scaler)
    elif type == 'NN':
        model = NeuralNetwork(args.clf, args.scaler)
    elif type == 'SVM':
        model = SVM(args.clf, args.scaler)
    else:
        raise Exception('(Error): Unknown model type.')

    df = pd.read_csv(args.data, sep=args.sep)
    y_pred = model.predict(df)
    print(y_pred)


if __name__ == '__main__':
    main()
