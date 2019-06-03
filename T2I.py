import argparse
from sys import version_info
from os import system
import numpy as np

from showImages import show
from loadFeatures import Features

user_input = input if version_info[0] > 2 else raw_input


def load(name):
    print("Loading %s ..." % (name))
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
    parser = argparse.ArgumentParser(description="Search image with tags in COCO")
    parser.add_argument('--name', type=str, default='CCA_0.npy',
                        help='Filename for the cca')
    args = parser.parse_args()

    cca = load(args.name)
    main(cca)
