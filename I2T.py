from sys import version_info, path
from os import system, environ

# APPEND CAFFE TO YOUR PATH
path.append("./caffe/python")  # export PYTHONPATH="/home/evann/dev/perso/Projet RecVis/caffe/python":$PYTHONPATH
environ['GLOG_minloglevel'] = '2'

import skimage.io as io
import matplotlib.pyplot as plt
import caffe
from loadFeatures import *
from CCA_search import imageToTagSearch

user_input = input if version_info[0] > 2 else raw_input


print("Loading Caffe model")
caffe_root = './caffe/'
model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
model_trained = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
layer_name = 'pool5/7x7_s1'

caffe.set_mode_cpu()

net = caffe.Net(model_prototxt, model_trained, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(mean_path).mean(1).mean(1))
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))  # if using RGB instead of BGR
transformer.set_raw_scale('data', 255.0)
net.blobs['data'].reshape(1, 3, 224, 224)


def urlToVec(image):
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))
    output = net.forward()
    imageVec = net.blobs[layer_name].data[0].reshape(1, -1)
    return imageVec[0]


def loadCCA(num=1, name='COCO'):
    print("Loading CCA %d (%s DB)..." % (num, name))
    load_features(name)
    imIds = get_images_id()
    W_T, W_V, phi_T, phi_V, D = np.load('Computed_CCA/CCA_{0}.npy'.format(num), encoding='latin1')
    return W_T, W_V, phi_T, phi_V, D, imIds


W_T, W_V, phi_T, phi_V, D, imIds = loadCCA()

url = ''
while(url != 'EXIT'):
    system("clear")
    url = user_input("Image URL: ")

    if (url[:2] == 'DB'):
        W_T, W_V, phi_T, phi_V, D, imIds = loadCCA(int(url[2:]))
    elif (url != 'EXIT'):
        image = io.imread(url)
        imageVec = urlToVec(url)
        tags, counts = imageToTagSearch(imageVec, W_V, D, 15, phi_T, W_T, imIds)
        print('\nCorresponding tags:')
        print(zip(tags, counts))
        plt.axis('off')
        plt.imshow(image)
        plt.show()
