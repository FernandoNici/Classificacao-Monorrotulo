from dado import Dado
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import  classification_report, confusion_matrix
from sklearn import metrics
import numpy as np
import pandas as pd
import string
from time import time
################################################################

def load_train_test():
    dataset = datasets.load_files('pt-BR/', encoding='utf-8', decode_error='ignore', shuffle=True)    

    train = Dado(dataset.data, dataset.target)
    categorias = dataset.target_names
    #18 labels
    return train, categorias

def classification():
    train, categorias = load_train_test()
    
    ngrama = 1
    import stem
    import tokenizer
    vectorizer_Tfidf = TfidfVectorizer(ngram_range=(1, ngrama), tokenizer=tokenizer.tokenize)
    #vectorizer_Tfidf = TfidfVectorizer(ngram_range=(1, ngrama), analyzer='char')
    X_new = vectorizer_Tfidf.fit_transform(train.X)
    dicionario = vectorizer_Tfidf.get_feature_names()
    
    k = 1000#10, 100, 1000
    print('nGrama: %d'%ngrama)
    print('Len %d'%len(dicionario))
    print('Aqui %d'%k)
    
    selector_ch2 = SelectKBest(chi2, k=k)
    X_new = selector_ch2.fit_transform(X_new, train.Y)
    #############################################
    
    np.seterr(divide='ignore')
    c_range = [10]#np.logspace(-3, 2, 6)#[1, 10, 100, 1000]
    gamma_range = [1]#np.logspace(-6, 2, 9)#[1.e-07, 1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]
    results = []
    for cinho in c_range:
        for gamm in gamma_range:
            print('C: %.6f'%cinho)
            print('gama: %.6f'%gamm)
            clf = SVC(tol=1e-3, C=cinho, gamma=gamm)

            predicted = cross_val_predict(clf, X_new, train.Y, cv=10)
            df_confusion = pd.crosstab(train.Y, predicted) 
            print(df_confusion.to_string())
            score = metrics.accuracy_score(train.Y, predicted)
            print("accuracy: %0.3f\n" % score)
            print(confusion_matrix(train.Y, predicted))
            print(metrics.classification_report(train.Y, predicted, target_names=categorias))
        

#MAIN###########################################################
classification()
