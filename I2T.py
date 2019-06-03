from sys import version_info
from os import system

import matplotlib.pyplot as plt
from urllib.request import urlopen
from PIL import Image

from loadFeatures import Features
import numpy as np

user_input = input if version_info[0] > 2 else raw_input


def load(name):
    print("Loading CCA %d ..." % (num))
    cca = np.load(name, encoding='latin1', allow_pickle=True).item()
    features = Features('features', 'val2017')
    cca.loadFeatures(features)
    return cca


def main(cca):
    url = ''
    while(url != 'EXIT'):
        system("clear")
        url = user_input("Image URL: ")
        image = Image.open(urlopen(url))
        tags, counts = cca.imageToTagSearch(image, 15)
        print('\nCorresponding tags:')
        print(list(zip(tags, counts)))
        plt.axis('off')
        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    num = 1
    cca = load('CCA_{0}.npy'.format(num))
    main(cca)
