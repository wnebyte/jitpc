import argparse
from decisiontree import DecisionTree
from knn import KNN
from naivebayes import NaiveBayes
from neuralnetwork import NeuralNetwork
from svm import SVM

"""
Train a new model

positional arguments:
  {DT,KNN,NB,NN,SVM}   model type
  DATA                 path to samples (.csv)

optional arguments:
  -h, --help           show this help message and exit
  -m MODEL_DIR         dir to save model
  -r REPORT_DIR        dir to save classification report
  -v {0,1,2,3,4}       verbosity
  -sep SEP             separator char for parsing csv data
"""

def main():
    parser = argparse.ArgumentParser(description='Train a new model')
    # positional arguments
    parser.add_argument('type', type=str,
                        choices=['DT', 'KNN', 'NB', 'NN', 'SVM'], help='model type')
    parser.add_argument('data', metavar='DATA', type=str, help='path to samples (.csv)')
    # optional arguments
    parser.add_argument('-m', type=str, dest='model_dir', help='dir to save model')
    parser.add_argument('-r', type=str, dest='report_dir', help='dir to save classification report')
    parser.add_argument('-v', type=int, dest='verbose', default=1,
                        choices=[0, 1, 2, 3, 4], help='verbosity')
    parser.add_argument('-sep', type=str, dest='sep', default=',', help='separator char for parsing csv data')
    args = parser.parse_args()
    run(args)

def run(args):
    type = args.type
    model = None

    if type == 'DT':
        model = DecisionTree()
    elif type == 'KNN':
        model = KNN()
    elif type == 'NB':
        model = NaiveBayes()
    elif type == 'NN':
        model = NeuralNetwork()
    elif type == 'SVM':
        model = SVM()
    else:
        raise Exception(str.format('(Error): Unknown model type: %s', type))

    model.train(args.data, sep=args.sep, model_out_dir=args.model_dir, report_out_dir=args.report_dir,
                verbose=args.verbose)


if __name__ == '__main__':
    main()
