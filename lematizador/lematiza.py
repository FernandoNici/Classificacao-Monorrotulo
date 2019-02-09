import subprocess

import os
from sklearn import datasets


rows = []
dataset = datasets.load_files('D:/pt-BR_300/', encoding='utf-8', decode_error='ignore', shuffle=False)

data = dataset.data
filenames = dataset.filenames
target_names = dataset.target_names
target = dataset.target

for index in range(len(filenames)):
    print(filenames[index])
    subprocess.call(['java', '-jar', 'Lematizador.jar' , filenames[index], 'nf'])
