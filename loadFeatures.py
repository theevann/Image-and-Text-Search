import numpy as np

featuresDir = 'Features'

dictionnary = None
imgIds = None
vecImages = None
vecWords = None
freqWords = None
annotations = None
wordsOfAnnotations = None
stopwords = None

cat2im = None
im2cat = None


def load_features(dbWordFeature):
    global vecImages, vecWords, freqWords, annotations, dictionnary, imgIds, wordsOfAnnotations, stopwords, cat2im, im2cat

    vecImages = np.load('{0}/imgFeatures.npy'.format(featuresDir), encoding='latin1').item()
    vecWords = np.load('{0}/wordFeatures_{1}.npy'.format(featuresDir, dbWordFeature), encoding='latin1').item()
    freqWords = np.load('{0}/wordFrequencies.npy'.format(featuresDir)).item()
    annotations = np.load('{0}/annotations.npy'.format(featuresDir)).item()
    wordsOfAnnotations = np.load('{0}/annotationsToWords.npy'.format(featuresDir)).item()
    stopwords = np.load('{0}/stopwords.npy'.format(featuresDir))

    cat2im = np.load('{0}/cat2im.npy'.format(featuresDir)).item()
    im2cat = np.load('{0}/im2cat.npy'.format(featuresDir)).item()

    dictionnary = list(vecWords.keys())
    imgIds = np.sort(list(vecImages.keys()))


def get_images_id():
    return imgIds


def get_categories():
    return cat2im.keys()


def get_visual_features():
    V = [(vecImages[im_ID])[0] for im_ID in imgIds]
    V = np.array(V)
    return V


def get_tag_features(fileName=None):
    if (fileName is not None):
        T = np.load(fileName)
    else:
        annot = [annotations[im_ID] for im_ID in imgIds]
        T = [sentencesToVec(ann) for ann in annot]
        T = np.array(T)
    return T


def sentenceToWords(sentence):
    words = sentence.lower().replace(',', '').replace('.', '').replace('"', '').replace("'s", '').split()
    words = [word for word in words if word not in stopwords and word in dictionnary]
    return words


def sentencesToWords(sentences):
    words = sentenceToWords(' '.join(sentences))
    return words


def annotationsToWords(imIds):
    if wordsOfAnnotations:
        words = np.concatenate([wordsOfAnnotations[imId] for imId in imIds])
    else:
        sentences = [' '.join(annotations[imId]) for imId in imIds]
        words = sentencesToWords(sentences)
    return words


def sentenceToVec(sentence):
    words = sentenceToWords(sentence)
    sentVec = np.zeros_like(vecWords['word'])
    sumFreq = 0
    for word in words:
        # weight = freqWords[word]
        weight = 1
        sentVec += vecWords[word] * weight
        sumFreq += weight
    if sumFreq != 0:
        sentVec /= sumFreq
    return sentVec


def sentencesToVec(sentences):
    vector = sentenceToVec(' '.join(sentences))
    return vector


def findImagesWithWordInAnnotations(word):
    ids = [imId for imId, words in wordsOfAnnotations.items() if word in words]
    return ids


def findImagesWithWordInCategory(word):
    ids = cat2im[word]
    return ids


# wordAnn = {imId : sentencesToWords(sentences) for imId, sentences in annotations.items()}
# np.save('wordsOfAnnotations', wordAnn)
