from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
import numpy as np
import tokenizer

################################################################
def classification():
    dataset = datasets.load_files('/home/pt-BR/', encoding='utf-8', decode_error='ignore', shuffle=False)

    #############################################
    # from sklearn import svm
    # clf = svm.SVC(cache_size=500)
    #############################################
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    #############################################
    # from sklearn.neural_network import MLPClassifier
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler(with_mean=False)
    # dataset.data = scaler.fit_transform(dataset.data)
    # clf = MLPClassifier(hidden_layer_sizes=(100,))
    #############################################

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        # ('scaler', StandardScaler(with_mean=False)),#just for mlp
        ('chiq', SelectKBest(chi2)),
        ('clf', clf)
    ])

    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        # 'tfidf__tokenizer': [tokenizer.tokenize], # just for analizer = word
        'tfidf__analyzer': ['char'],
        # 'tfidf__analyzer': ['word'],
        'chiq__k': [1000], #[10,100,1000]
        # 'clf__kernel': ['rbf'], # or ['linear'] just for SVM
        # 'clf__gamma': np.logspace(-2, 1, 4), #RBF
        # 'clf__C': [10, 100] #[10,100,1000]
        'clf__n_estimators': [1000], # Random Forest
        'clf__max_depth': [5],
        'clf__verbose': [50],
        # 'clf__solver': ['adam'], #mlp
        # 'clf__alpha':  np.logspace(-2, 1, 4) #é do mlp
    }

    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    scores = ['f1_weighted']  # or 'samples'
    #############################################
    for score in scores:
        np.seterr(divide='ignore')
        grid = GridSearchCV(pipeline, parameters, n_jobs=2, scoring=score, verbose=50, cv=10)
        grid.fit(dataset.data, dataset.target)
        print(grid.best_estimator_)

        print("The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))

        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print("Detailed classification:")
        print()
        y_true, y_pred = dataset.target, grid.predict(dataset.data)
        print(classification_report(dataset.target, y_pred))

    print()

    #############################################


###########################################################
classification()