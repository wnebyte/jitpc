from classifier import BaseClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from utils import get, mk_filepath

class DecisionTree(BaseClassifier):
    """
    A class used to represents a trainable, binary decision tree classifier capable of making predictions on
    samples of a particular dataset.

    Methods
    -------
    train(dataset_url, sep=',', name=None, model_out_dir=None, report_out_dir=None, verbose=0)
        Trains a new classifier.
    predict(X)
        Predicts binary class values for X.
    """

    def __init__(self, clf_file=None, scaler_file=None):
        """Constructs a new instance.

        :param clf_file: a .pkl file from where to load a classifier; default=None.
        :param scaler_file: a .pkl file from where to load a scaler; default=None.
        """
        super().__init__(clf_file, scaler_file)

    def train(self, dataset_url, sep=',', name=None, model_out_dir=None, report_out_dir=None, verbose=0):
        """Trains a new decision tree classifier.

        The classifier is trained using the following steps:
        1. Dataset is loaded.
        2. Data is prepared/pre-processed.
        3. Data is oversampled using SMOTE.
        4. Data is split into training and test sets.
        5. Data is standardized using a standard scaler.
        6. The best estimator is found using cross-validation.
        7. The model and classification report is saved to disk (optional).

        :param dataset_url: a url from where a csv-formatted dataset can be loaded.
        :param sep: a separator char used during the parsing of the csv data; default=','.
        :param name: a name for the serialized model/scaler/classification report; default=None.
        :param model_out_dir: a directory where the model is to be saved to; default=None;
        if set to None the model will not be saved.
        :param report_out_dir: a directory where the classification report is to be saved to; default=None;
        if set to None the classification report will not be saved.
        :param verbose: controls the verbosity: the higher, the more messages; default=3;
        > 0. the best params and the classification report is displayed;
        > 1. the cv computation time for each fold and parameter candidate is also displayed;
        > 2. the cv score is also displayed;
        > 3. the cv fold and candidate parameter indexes are also displayed together with the starting time of the
        computation.
        """
        # Step 1: Load dataset
        df = self.__load_data__(dataset_url, sep)
        # Step 2: Prepare dataset
        x, y = self.__prepare_data__(df)
        # Step 3: Oversample dataset
        x, y = self.__oversample_data__(x, y)
        # Step 4: Split dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.2,
                                                            random_state=123,
                                                            stratify=y)
        # Step 5: Standardize data
        scaler = StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        # Step 5: Find best estimator using cross-validation
        hyperparams = [
            {
                #'criterion': ['gini', 'entropy', 'log_loss'],
                'criterion': ['log_loss'],
                'splitter': ['best']
            }
        ]
        clf = GridSearchCV(DecisionTreeClassifier(), hyperparams, verbose=verbose)
        clf.fit(x_train_scaled, y_train)
        # Step 6: Save classification report and model
        y_pred = clf.predict(x_test_scaled)
        report = classification_report(y_test, y_pred)
        if verbose > 0:
            print(clf.best_params_)
            print(report)
        if report_out_dir is not None:
            self.__dump_classification_report__(
                report, mk_filepath(report_out_dir, get('dt', name), 'report', dataset_url, '.txt'))
        if model_out_dir is not None:
            self.__dump_model__(
                clf, mk_filepath(model_out_dir, get('dt', name), 'clf', dataset_url, '.pkl'))
            self.__dump_scaler__(
                scaler, mk_filepath(model_out_dir, get('dt', name), 'scaler', dataset_url, '.pkl'))
        self.clf = clf
        self.scaler = scaler
