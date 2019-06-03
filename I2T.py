import argparse
from os import system
from sys import version_info
from urllib.request import urlopen

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from loadFeatures import Features

user_input = input if version_info[0] > 2 else raw_input

def load(name):
    print("Loading %s ..." % (name))
    cca = np.load(name, encoding='latin1', allow_pickle=True).item()
    features = Features('features', 'train2017')
    cca.loadFeatures(features)
    return cca


def main(cca):
    url = ''
    while(url != 'EXIT'):
        system("clear")
        url = input("Image URL: ")
        image = Image.open(urlopen(url))
        tags, counts = cca.imageToTagSearch(image, 15)
        print('\nCorresponding tags:')
        print(list(zip(tags, counts)))
        plt.axis('off')
        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search for tags given image using CCA")
    parser.add_argument('--name', type=str, default='CCA_0.npy',
                        help='Filename for the cca')
    args = parser.parse_args()

    cca = load(args.name)
    main(cca)
