from decisiontree import DecisionTree
from knn import KNN
from naivebayes import NaiveBayes
from neuralnetwork import NeuralNetwork
from svm import SVM

def main():
    dataset_url = '../res/data/qt_metrics.csv'
    models = [DecisionTree(), KNN(), NaiveBayes(), NeuralNetwork(), SVM()]

    for model in models:
        model.train(dataset_url, verbose=3)


if __name__ == '__main__':
    main()
