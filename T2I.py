from sys import version_info
from os import system
from showImages import show
from loadFeatures import *
from CCA_search import textToImageSearch

user_input = input if version_info[0] > 2 else raw_input


def load(num=1, name='COCO'):
    print("Loading CCA %d (%s DB)..." % (num, name))
    load_features(name)
    imIds = get_images_id()
    W_T, W_V, phi_T, phi_V, D = np.load('Computed_CCA/CCA_{0}.npy'.format(num), encoding='latin1')
    return W_T, W_V, phi_T, phi_V, D, imIds


W_T, W_V, phi_T, phi_V, D, imIds = load()

search = ''
while(search != 'EXIT'):
    system("clear")
    search = user_input("Search Terms: ")

    if (search[:2] == 'DB'):
        W_T, W_V, phi_T, phi_V, D, imIds = load(int(search[2:]))
    elif (search != 'EXIT'):
        res_IDs, similarities = textToImageSearch(search, W_T, D, 10, phi_V, W_V, imIds)
        show(res_IDs.tolist())
