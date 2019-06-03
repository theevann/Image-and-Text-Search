from sys import version_info
from os import system

from showImages import show
from loadFeatures import Features
import numpy as np

user_input = input if version_info[0] > 2 else raw_input


def load(name):
    print("Loading CCA %d ..." % (num))
    cca = np.load(name, encoding='latin1', allow_pickle=True).item()
    features = Features('features', 'train2017')
    cca.loadFeatures(features)
    return cca


def main(cca):
    search = ''
    while(search != 'EXIT'):
        system("clear")
        search = user_input("Search Terms: ")
        res_IDs, similarities = cca.textToImageSearch(search, 5)
        print(similarities)
        show(res_IDs.tolist())


if __name__ == '__main__':
    num = 0
    cca = load('CCA_{0}.npy'.format(num))
    main(cca)
