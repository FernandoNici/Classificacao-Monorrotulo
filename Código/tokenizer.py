import nltk

stopwords = nltk.corpus.stopwords.words("portuguese")
caracteres_especiais = {'–', ';', ':', '=', '…', '•', '”', '’', '‘', '—', '~', '“', '<', '>', '+', '-', '/', '\\', '!',
                        '?', '@', '#', '$', '%', '&', '*', ',', '(', ')', '.', '{', '}', '[', ']', '´', '`', '0', '1',
                        '2', '3', '4', '5', '6', '7', '8', '9'}
stop_custom = ['łukaszewicz', 'カードキャプターさくら', 'バーンディ', 'ワールド', '爵迹']
stopwords.extend(stop_custom)

stemmer = nltk.stem.RSLPStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        go_append = True

        for caract in caracteres_especiais:
            if item.find(caract) >= 0:
                go_append = False

        if item not in caracteres_especiais and go_append:
            item = item.replace("'", "")
            if item not in stopwords and len(item) > 1:
                raiz = stemmer.stem(item)
                stemmed.append(raiz)

    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems