# -*- coding: utf-8 -*-

from CCATextImage import CCATextImage
from loadFeatures import Features

import numpy as np
import argparse


parser = argparse.ArgumentParser(description="Compute CCA for COCO")
parser.add_argument('--dim', type=int, default=80,
                    help='evaluation interval (default: 80)')
parser.add_argument('--output', type=str, default='CCA_0',
                    help='set the output filename for the cca')
args = parser.parse_args()


# LOAD FEATURES
print("Loading features")
features = Features("features", "train2017")


# COMPUTE COVARIANCE AND SOLVE CCA
print("Computing CCA")
cca = CCATextImage(features, dimension=args.dim)
cca.solve()


# SAVE CCA
cca.unloadFeatures()
np.save(args.output, cca)
