from abc import ABC, abstractmethod
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

class Classifier(ABC):
    """
    An abstract class used to represents a trainable machine-learning model capable of making predictions.

    Methods
    -------
    @abstractmethod
    __load_data__(dataset_url, sep=',')
        Reads and returns data.
    @abstractmethod
    __prepare_data__(df)
        Pre-processes data.
    @abstractmethod
    __oversample_data__(x, y)
        Oversamples/resamples data.
    @abstractmethod
    __standardize_data__(x_train, x_test)
        Standardizes training and test data.
    @abstractmethod
    train(dataset_url, sep=',', name=None, model_out_dir=None, report_out_dir=None, verbose=0)
        Trains a new model.
    @abstractmethod
    predict(X)
        Predicts values for X.
    """

    @abstractmethod
    def __load_data__(self, dataset_url, sep=','):
        pass

    @abstractmethod
    def __prepare_data__(self, df):
        pass

    @abstractmethod
    def __oversample_data__(self, x, y):
        pass

    @abstractmethod
    def __standardize_data__(self, x_train, x_test):
        pass

    @abstractmethod
    def train(self, dataset_url, sep=',', name=None, model_out_dir=None, report_out_dir=None, verbose=0):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class BaseClassifier(Classifier):
    """
    An abstract base class used to represents a trainable, binary classification model capable of making predictions on
    samples of a particular dataset.

    Attributes
    ----------
    clf : sklearn.Classifier
        a classifier.
    scaler : sklearn.StandardScaler
        a scaler.

    Methods
    -------
    __load_data__(dataset_url, sep=',')
        Reads and returns the data located at dataset_url
    __prepare_data__(df)
        Pre-processes data.
    __oversample_data__(x, y)
        Oversamples data.
    __standardize_data__(x_train, x_test)
        Standardizes training and test data.
    __dump_model__(clf, model_file)
        Saves model to disk.
    __dump_scaler__(scaler, scaler_file)
        Saves scaler to disk.
    __dump_classification_report__(report, txt_file)
        Saves report to disk.
    display(df, rows=10, columns=32)
        Prints contents of dataframe.
    train(dataset_url, sep=',', name=None, model_out_dir=None, report_out_dir=None, verbose=0)
        Trains a new classifier.
    predict(X)
        Predicts class values for X.
    """

    def __init__(self, clf_file=None, scaler_file=None):
        """Constructs a new instance.

        :param clf_file: a .pkl file containing a classifier to be loaded; default=None.
        :param scaler_file: a .pkl file containing a scaler to be loaded; default=None.
        """
        if clf_file is None:
            self.clf = None
        else:
            self.clf = joblib.load(clf_file)
        if scaler_file is None:
            self.scaler = None
        else:
            self.scaler = joblib.load(scaler_file)

    def __load_data__(self, dataset_url, sep=','):
        """Reads and returns the csv-formatted dataset located the specified url.

        :param dataset_url: a url.
        :param sep: a separator char used in the parsing of the corresponding dataset.
        :return: a pandas dataframe containing the contents of the corresponding dataset.
        """
        df = pd.read_csv(dataset_url, sep=sep)
        return df

    def __prepare_data__(self, df, predict=False):
        """Pre-processes the specified dataset by:

        1. Dropping unwanted features from the dataset (if present).
        2. Replacing all NaN values in the dataset with 0s.
        3. Applying a transformation on the bugcount feature by setting its value to False if its 0,
        and to True if it's not 0 (if predict=False).
        4. Splitting the dataset into two vectors where the bugcount feature is dropped from the 2nd vector.

        :param df: a pandas dataframe.
        :return: a tuple containing two vectors if predict=False,
        otherwise a tuple containing only one vector.
        """
        if 'commit_id' in df.columns:
            df = df.drop('commit_id', axis=1)
        if 'author_date' in df.columns:
            df = df.drop('author_date', axis=1)
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        imputed = imp.fit_transform(df)
        df = pd.DataFrame(imputed, columns=df.columns)
        if predict is False:
            # branch taken during training
            df.loc[df['bugcount'] == 0, 'bugcount'] = False
            df.loc[df['bugcount'] != 0, 'bugcount'] = True
            df = df.infer_objects()
            y = df.bugcount
            x = df.drop('bugcount', axis=1)
            return (x, y)
        else:
            # branch taken during prediction
            if 'bugcount' in df.columns:
                df = df.drop('bugcount', axis=1)
            x = df.infer_objects()
            return (x, None)

    def __oversample_data__(self, x, y):
        """Oversamples/resamples the specified data using SMOTE.

        Consequently each possible bugcount label (True, False) will occur an even number of times in the returned
        data.

        :param x: matrix containing the data which is to be sampled.
        :param y: corresponding label for each sample in x.
        :return: tuple containing the resampled data.
        """
        oversample = SMOTE()
        x, y = oversample.fit_resample(x, y)
        return (x, y)

    def __standardize_data__(self, x_train, x_test):
        """Standardizes the specified training and test data.

        :param x_train: training data to be scaled.
        :param x_test: test data to be scaled.
        :return: a tuple containing the standardized training and test data.
        """
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        return (x_train_scaled, x_test_scaled)

    def train(self, dataset_url, sep=',', name=None, model_out_dir=None, report_out_dir=None, verbose=0):
        """Abstract method to be implemented by subclasses.

        :param dataset_url: a url from where a csv-formatted dataset can be read.
        :param sep: a separator char used during the reading of the csv data; default=','.
        :param name: a partial name for the serialized model/scaler/classification report; default=None.
        :param model_out_dir: a directory where the model and scaler is to be saved; default=None.
        :param report_out_dir: a directory where the classification report is to be saved; default=None.
        :param verbose: controls the verbosity: the higher, the more messages; default=0;
        """
        pass

    def predict(self, X):
        """Predicts class value for X.

        :param X: the input samples.
        :return: the predicted class for each sample in X is returned.
        :raises: Exception if this object's classifier and or scaler has not yet been initialized.
        """
        if self.clf is None or self.scaler is None:
            raise Exception('This classification model has not yet been trained.')
        x, y = self.__prepare_data__(X, predict=True)
        x_scaled = self.scaler.transform(x)
        return self.clf.predict(x_scaled)

    def __dump_model__(self, clf, model_file):
        """Saves the specified classifier to disk.

        :param clf: a classifier/model.
        :param model_file: a .pkl file where the specified classifier is to be serialized.
        """
        joblib.dump(clf, model_file)

    def __dump_scaler__(self, scaler, scaler_file):
        """Saves the specified scaler to disk.

        :param scaler: a scaler.
        :param scaler_file: a .pkl file where the specified scaler is to be serialized.
        """
        joblib.dump(scaler, scaler_file)

    def __dump_classification_report__(self, report, txt_file):
        """Saves the specified classification report to disk.

        :param report: a report.
        :param txt_file: a .txt file where the specified classification is to be serialized.
        """
        with open(txt_file, 'w') as f:
            f.write(report)

    def display(self, df, rows=10, columns=32):
        """Prints the contents of the specified dataframe to stdout.

        :param df: a dataframe.
        :param rows: max number of rows that should be displayed.
        :param columns: max number of columns that should be displayed.
        """
        with pd.option_context('display.max_rows', rows,
                               'display.max_columns', columns,
                               'display.precision', 3):
            print(df)
