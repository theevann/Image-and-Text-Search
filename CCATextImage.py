# -*- coding: utf-8 -*-
from CCA import CCA

import numpy as np
import torch


class CCATextImage(CCA):
    """docstring for CCATextImage."""
    def __init__(self, features, dimension, regularization=1, power=-1):
        super(CCATextImage, self).__init__(dimension, regularization, power)
        self.features = features

    def loadFeatures(self, features):
        self.features = features

    def unloadFeatures(self):
        self.features = None

    def solve(self):
        T = self.features.getTagFeatures()
        V = self.features.getVisualFeatures()[:, :2000]
        return super().solve([T, V])

    def textToImageSearch(self, text, n_images):
        phi_t = self.features.sentenceToVec(text)
        similarities, idx = self.getSimilarities(phi_t, 0, 1)
        if n_images > 0:
            idx = idx[:n_images]
        return self.features.imgIds[idx], similarities[idx]

    def imageToTagSearch(self, image, n_tags):
        phi_v = self.features.imageToVec(image)[:2000]
        print(phi_v)
        similarities, idx = self.getSimilarities(phi_v, 1, 0)
        ids = self.features.imgIds[idx[:50]]
        # from showImages import show
        # for id in ids:
        #     show(id)
        return self.features.mostCommonWordsIn(ids, n_tags)


    def tagsToImageSearch(tags, W_T, D, n_images, phi_V, W_V, img_ids):
        projected_V = np.dot(phi_V, W_V)
        scaled_proj_V = np.dot(projected_V, D)
        norm_scaled_proj_V = np.linalg.norm(scaled_proj_V, axis=1)

        T = np.array([sentenceToVec(tag) for tag in tags])
        phi_T = np.array([mapTagFeatures(t) for t in T])
        projected_T = np.dot(phi_T, W_T)
        scaled_proj_T = np.dot(projected_T, D)
        scaled_proj_T = scaled_proj_T.astype(float)
        norm_scaled_proj_T = np.linalg.norm(scaled_proj_T, axis=1)

        prods = np.outer(norm_scaled_proj_T, norm_scaled_proj_V)
        dots = np.dot(scaled_proj_T, scaled_proj_V.T)
        similarities = dots / prods

        idx = similarities.argsort(axis=1)[:, ::-1]
        sorted_similarities = np.sort(similarities, axis=1)[:, ::-1]
        if n_images > 0:
            idx = idx[:, :n_images]
            sorted_similarities = sorted_similarities[:, :n_images]
        return np.take(img_ids, idx), sorted_similarities

    def imagesToTagSearch(visual_features, W_V, D, n_tags, phi_T, W_T, img_ids):
        projected_T = np.dot(phi_T, W_T)
        scaled_proj_T = np.dot(projected_T, D)
        norm_scaled_proj_T = np.linalg.norm(scaled_proj_T, axis=1)

        phi_V = np.array([mapVisualFeatures(v) for v in visual_features])
        projected_V = np.dot(phi_V, W_V)
        scaled_proj_V = np.dot(projected_V, D)
        norm_scaled_proj_V = np.linalg.norm(scaled_proj_V, axis=1)

        tags = []

        prods = np.outer(norm_scaled_proj_V, norm_scaled_proj_T)
        dots = np.dot(scaled_proj_V, scaled_proj_T.T)
        similarities = dots / prods

        idx = similarities.argsort(axis=1)[:, ::-1][:, :50]
        ids = img_ids[idx]

        for idsImg in ids:
            words = annotationsToWords(idsImg)
            tags.append([tag for tag, it in Counter(words).most_common(n_tags)])

        return tags
