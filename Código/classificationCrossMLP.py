from dado import Dado
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
################################################################
def load_train():
    dataset = datasets.load_files('pt-BR/', encoding='utf-8', decode_error='ignore', shuffle=True)    
    train = Dado(dataset.data, dataset.target)
    categorias = dataset.target_names
    #18 labels
    return train, categorias


def benchmark(clf, X_new, y_train, target_names):
    predicted = cross_val_predict(clf, X_new, y_train, cv=10)
    df_confusion = pd.crosstab(y_train, predicted) 
    print(df_confusion.to_string())
    
    score = metrics.accuracy_score(y_train, predicted)
    print("accuracy: %0.3f\n" % score)
    print(metrics.classification_report(y_train, predicted, target_names=target_names))
    
def classification():
    train, categorias = load_train()
    ngrama = 1

    import tokenizer
    vectorizer_Tfidf = TfidfVectorizer(ngram_range=(1, ngrama), tokenizer=tokenizer.tokenize)
    #vectorizer_Tfidf = TfidfVectorizer(ngram_range=(1, ngrama), analyzer='char')
    X_new = vectorizer_Tfidf.fit_transform(train.X)

    scaler = StandardScaler(with_mean=False)
    X_new = scaler.fit_transform(X_new)

    k = 1000#10, 100, 1000
    dicionario = vectorizer_Tfidf.get_feature_names()
    print('nGrama: %d'%ngrama)
    print('Len %d'%len(dicionario))
    print('Chi %d'%k)
    
    selector_ch2 = SelectKBest(chi2, k=k)
    X_new = selector_ch2.fit_transform(X_new, train.Y)
    ############################################
    
    alphas = [4.64]#np.logspace(-5, 3, 5)
    solver_range = ['adam']#'lbfgs', 
    np.seterr(divide='ignore')
    for al in alphas:
        for sol in solver_range:
            print('alpha: %.5f'%al)
            print('Solver: %s'%sol)
            clf = MLPClassifier(solver=sol, alpha=al, hidden_layer_sizes=(100,), random_state=1)
            benchmark(clf, X_new, train.Y, categorias)


#MAIN###########################################################
classification()
