# -*- coding: utf-8 -*-

from CCATextImage import CCATextImage
from loadFeatures import Features
import numpy as np


# LOAD FEATURES
print("Loading features")
features = Features("features", "train2017")


# COMPUTE COVARIANCE AND SOLVE CCA
print("Computing CCA")
cca = CCATextImage(features, dimension=80)
cca.solve()


# SAVE CCA
cca.unloadFeatures()
np.save('CCA_0', cca)
