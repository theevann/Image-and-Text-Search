import numpy as np
from pycocotools.coco import COCO
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe

# export PYTHONPATH="/home/evann/Documents/Dropbox/Travail/MVA/Object Recognition/Projet/caffe-master/python":$PYTHONPATH
# export PYTHONPATH="/home/evann/dev/perso/Projet RecVis/caffe-master/python":$PYTHONPATH

def log(s):
    if (logging):
        print(s)

logging = True
batchSize = 10

log("Loading images names through COCO api")
dataDir = '/media/evann/Data/MS COCO/'
# dataType = 'train2014'
dataType = 'val2014'
annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
coco = COCO(annFile)
imgs = coco.loadImgs(coco.getImgIds())
imgsNames = [img['file_name'] for img in imgs]
imgsIds = [img['id'] for img in imgs]
nbImages = len(imgs)
log("%d images names loaded" % nbImages)


log("Defining caffe variables")
# Main path to your caffe installation
caffe_root = './caffe-master/'
model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
model_trained = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
layer_name = 'pool5/7x7_s1'


# Setting this to CPU
caffe.set_mode_cpu()


log("Loading and initializing caffe model")
# Loading the Caffe model, setting preprocessing parameters
net = caffe.Net(model_prototxt, model_trained, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(mean_path).mean(1).mean(1))
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))  # if using RGB instead of BGR
transformer.set_raw_scale('data', 255.0)
imgsVectors = {}


def oneByOne():
    net.blobs['data'].reshape(1, 3, 224, 224)

    for i in range(nbImages):
        log("Processing image %d" % (i+1))
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(dataDir + dataType + '/' + imgsNames[i]))
        output = net.forward()
        imgsVectors[imgsIds[i]] = net.blobs[layer_name].data[0].reshape(1, -1).copy()


def nByn(n):
    net.blobs['data'].reshape(n, 3, 224, 224)

    batches = int(nbImages / n)
    for i in range(batches):
        log("Processing batch %d over %d" % (i+1, batches+1))

        for j in range(n):
            net.blobs['data'].data[j, ...] = transformer.preprocess('data', caffe.io.load_image(dataDir + dataType + '/' + imgsNames[j + i*n]))
        output = net.forward()
        for j in range(n):
            imgsVectors[imgsIds[j + i*n]] = net.blobs[layer_name].data[j].reshape(1, -1).copy()

    log("Processing batch %d over %d" % (batches+1, batches+1))
    for j in range(nbImages - batches*n):
        net.blobs['data'].data[j, ...] = transformer.preprocess('data', caffe.io.load_image(dataDir + dataType + '/' + imgsNames[j + batches*n]))
    output = net.forward()
    for j in range(nbImages - batches*n):
        imgsVectors[imgsIds[j + batches*n]] = net.blobs[layer_name].data[j].reshape(1, -1).copy()


log("Loading and Running CNN over %d images" % nbImages)
nByn(batchSize)

log("Saving images features...")
np.save('imgsFeatures_%s' % dataType, imgsVectors)

## {str(key).zfill(6): value for key, value in imgsVectors.items()}

# # Labels
# imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
# label_mapping = np.loadtxt(imagenet_labels, str, delimiter='\t')

# for i in range(nbImages):
#     best_n = net.blobs['prob'].data[i].flatten().argsort()[-1:-6:-1]
#     print('\n', imageNames[i], label_mapping[best_n])

# import matplotlib.pyplot as plt
# I = plt.imread(image_path)
# plt.imshow(I)
# plt.show()
