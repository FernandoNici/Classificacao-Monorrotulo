# Classificacao-Monorrotulo
Classificação monorrótulo de gêneros cinematográficos

## Referências  
PORTOLOSE  G.; FELTRIM V. D. On the Use of Synopsis-based Features for Film Genre Classification. Anais do 15º Encontro Nacional de Inteligência Artificial e Computacional (ENIAC 2018), 2018. p. 892-902.

## About
<img src="https://github.com/FernandoNici/Classificacao-Monorrotulo/blob/master/312x276-primary-blue.png" alt="logo" height="50" width="50">  Base utilizada: dataset de sinopses em português de filmes disponibilizada pela The Movie Database (TMDb) <https://www.themoviedb.org/>

## Ambiente _Linux 64bits_ utilizando Pycharm:

**OBS:** exportação da configuração está no arquivo [settings](https://github.com/FernandoNici/Classificacao-Monorrotulo/blob/master/Pycharm/settings.jar)

  * NumPy - array processing for numbers, strings, records an objects
  >- Version: 1.15.4
  >- numpy.org

  * SciPy - a Python-based ecosystem of open-source software for mathematics, science, and engineering.
  >- Version: 1.1.0
  >- scipy.org

  * NLTK - Natural Language Toolkit 
  >- Version: 3.4 
  >- nltk.org

  * scikit-learn - A set of puthon modules for ML and data mining
  >- Version: 0.20.1
  >- scikit-learn.org
  
  
## Ambiente _Windows 64bits_

  * astroid           1.6.5
  * colorama          0.3.9
  * cycler            0.10.0
  * isort             4.3.4
  * kiwisolver        1.0.1
  * lazy-object-proxy 1.3.1
  * mccabe            0.6.1
  * nltk              3.3
  * numpy             1.14.3
  * pandas            0.22.0
  * pip               10.0.1
  * pylint            1.9.2
  * pyparsing         2.2.0
  * python-dateutil   2.7.2
  * pytz              2018.4
  * scikit-learn      0.19.1
  * scipy             1.1.0
  * setuptools        39.0.1
  * six               1.11.0
  * wrapt             1.10.11

## Instalando o pacote de stopwords, rslp e punkt do nltk
<pre>
import nltk
nltk.download()
PyDev console: starting.
Python 3.5.2 (default, Nov 23 2017, 16:37:01) 

NLTK Downloader
---------------------------------------------------------------------------
    d) Download   l) List    u) Update   c) Config   h) Help   q) Quit
---------------------------------------------------------------------------
Escolha a opção 'd'; 
Em seguida digite: stopwords
Em seguida digite: rslp -- removedor de sufixo da língua portuguesa
Em seguida digite: punkt -- separa palavras tanto por espaço quanto pontuação

por fim 'q' para sair.
---------------------------------------------------------------------------
import nltk
nltk.corpus.stopwords.words("portuguese")
stemmer = nltk.stem.RSLPStemmer()
stemmer.stem('terminar')
'termin'
</pre>
