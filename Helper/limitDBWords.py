import numpy as np

newDBName = 'NEW_DB'

featuresDir = 'Features'
vecWordsCOCO = np.load('{0}/wordFeatures_{1}.npy'.format(featuresDir, "COCO")).item()
vecWordsNEW = np.load('{0}/wordFeatures_{1}.npy'.format(featuresDir, newDBName)).item()

dictionnary = vecWordsCOCO.keys()
vecWords_NEW_LIMITED = {word: vecWordsNEW[word] for word in dictionnary if word in vecWordsNEW}

np.save('{0}/wordFeatures_{1}_LIMITED.npy'.format(featuresDir, newDBName), vecWords_NEW_LIMITED)
