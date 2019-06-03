from Helper.extractImageFeatures import Transformer
from torchvision import models

from nltk.corpus import stopwords as sw
from collections import Counter
import numpy as np
import torch
import re


class Features():
    """docstring for Features."""
    def __init__(self, featuresDir, dataType):
        super(Features, self).__init__()
        vecImages = torch.load('{0}/imgFeatures_{1}'.format(featuresDir, dataType))
        vecAnnotations = np.load('{0}/annotationFeatures_{1}.npy'.format(featuresDir, dataType), allow_pickle=True).item()

        self.vecWords = np.load('{0}/wordFeatures.npy'.format(featuresDir), encoding='latin1', allow_pickle=True).item()
        self.freqWords = np.load('{0}/wordFrequencies.npy'.format(featuresDir), allow_pickle=True).item()
        self.annotations = np.load('{0}/annotations_{1}.npy'.format(featuresDir, dataType), allow_pickle=True).item()
        self.ann2words = np.load('{0}/annotation2word_{1}.npy'.format(featuresDir, dataType), allow_pickle=True).item()
        self.stopwords = sw.words("english")

        self.dictionnary = list(self.vecWords.keys())
        self.imgIds = np.sort(list(vecImages.keys()))

        self.imageMatrix = torch.stack([vecImages[imId] for imId in self.imgIds])
        self.annotationMatrix = torch.stack([vecAnnotations[imId] for imId in self.imgIds])

        self.cat2im = np.load('{0}/cat2im_{1}.npy'.format(featuresDir, dataType), allow_pickle=True).item()
        self.im2cat = np.load('{0}/im2cat_{1}.npy'.format(featuresDir, dataType), allow_pickle=True).item()

        self.imageModel = None
        self.transformer = None

    def getVisualFeatures(self):
        return self.imageMatrix

    def getTagFeatures(self):
        return self.annotationMatrix


    def findImagesFromWordInAnnotations(self, word):
        ids = [imId for imId, words in self.ann2words.items() if word in words]
        return ids

    def findImagesInCategory(self, cat):
        return self.cat2im[cat]

    def mostCommonWordsIn(self, ids, n_tags):
        words = [word for id in ids for word in self.ann2words[id]]
        most_commons = Counter(words).most_common(n_tags)
        return list(zip(*most_commons))


    def sentenceToWords(self, sentence):
        words = re.sub('[\.,"]|\'s', '', sentence.lower()).split()
        return [word for word in words if word not in self.stopwords and word in self.dictionnary]

    def sentencesToWords(self, sentences):
        return self.sentenceToWords(' '.join(sentences))

    def sentenceToVec(self, sentence):
        return np.sum(self.vecWords[word] for word in self.sentenceToWords(sentence))

    def sentencesToVec(self, sentences):
        return self.sentenceToVec(' '.join(sentences))


    def imageToVec(self, image):
        if not self.imageModel:
            self.loadImageModel()
        imageTensor = self.transformer(image)
        # import ipdb; ipdb.set_trace()
        import numpy as np
        print(np.array(image))
        print(imageTensor)
        # import matplotlib.pyplot as plt
        # plt.axis('off')
        # plt.imshow(imageTensor.permute(1,2,0).numpy())
        # plt.show()
        output = self.imageModel(imageTensor.unsqueeze(0)).squeeze()
        return output

    def loadImageModel(self):
        resnet50 = models.resnet50(pretrained=True)
        self.imageModel = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
        self.transformer = Transformer()
