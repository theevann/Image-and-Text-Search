# -*- coding: utf-8 -*-
from loadFeatures import annotationsToWords, sentenceToVec
from CCA_utils import mapTagFeatures, mapVisualFeatures
from collections import Counter
import numpy as np


def similarity(scaled_proj_x, scaled_proj_y):
    norm_prod = np.linalg.norm(scaled_proj_x) * np.linalg.norm(scaled_proj_y)
    sim = np.dot(scaled_proj_x, scaled_proj_y.T) / norm_prod
    return sim


def textToImageSearch(text, W_T, D, n_images, phi_V, W_V, img_IDs):
    projected_V = np.dot(phi_V, W_V)
    scaled_proj_V = np.dot(projected_V, D)
    norm_scaled_proj_V = np.linalg.norm(scaled_proj_V, axis=1)

    t = sentenceToVec(text)
    phi_t = mapTagFeatures(t)
    projected_t = np.dot(phi_t, W_T)
    scaled_proj_t = np.dot(projected_t, D)
    norm_scaled_proj_t = np.linalg.norm(scaled_proj_t)

    prods = norm_scaled_proj_t * norm_scaled_proj_V
    dots = np.dot(scaled_proj_t, scaled_proj_V.T)
    similarities = dots / prods

    idx = similarities.argsort()[::-1]
    if n_images > 0:
        idx = idx[:n_images]

    return img_IDs[idx], similarities[idx]


def tagsToImageSearch(tags, W_T, D, n_images, phi_V, W_V, img_IDs):
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
    return np.take(img_IDs, idx), sorted_similarities


def imageToTagSearch(visual_feature, W_V, D, n_tags, phi_T, W_T, img_IDs):
    projected_T = np.dot(phi_T, W_T)
    scaled_proj_T = np.dot(projected_T, D)
    norm_scaled_proj_T = np.linalg.norm(scaled_proj_T, axis=1)

    phi_v = mapVisualFeatures(visual_feature)
    projected_v = np.dot(phi_v, W_V)
    scaled_proj_v = np.dot(projected_v, D)
    norm_scaled_proj_v = np.linalg.norm(scaled_proj_v)

    prods = norm_scaled_proj_v * norm_scaled_proj_T
    dots = np.dot(scaled_proj_v, scaled_proj_T.T)
    similarities = dots / prods

    idx = similarities.argsort()[::-1][:50]
    ids = img_IDs[idx]
    words = annotationsToWords(ids)

    mostCommons = Counter(words).most_common(n_tags)
    tags = [tag for tag, it in mostCommons]
    counts = [it for tag, it in mostCommons]
    return tags, counts


def imagesToTagSearch(visual_features, W_V, D, n_tags, phi_T, W_T, img_IDs):
    projected_T = np.dot(phi_T, W_T)
    scaled_proj_T = np.dot(projected_T, D)
    norm_scaled_proj_T = np.linalg.norm(scaled_proj_T, axis=1)

    phi_V = np.array([mapVisualFeatures(v) for v in visual_features])
    projected_V = np.dot(phi_V, W_V)
    scaled_proj_V = np.dot(projected_V, D)
    norm_scaled_proj_V = np.linalg.norm(scaled_proj_V, axis=1)

    tags = []
    try:
        prods = np.outer(norm_scaled_proj_V, norm_scaled_proj_T)
        dots = np.dot(scaled_proj_V, scaled_proj_T.T)
        similarities = dots / prods

        idx = similarities.argsort(axis=1)[:, ::-1][:, :50]
        ids = img_IDs[idx]

        for idsImg in ids:
            words = annotationsToWords(idsImg)
            tags.append([tag for tag, it in Counter(words).most_common(n_tags)])
    except:
        for i in np.arange(len(visual_features) / 200 + 1):
            print str(i) + ' / ' + str(len(visual_features) / 200 + 1)
            prods = np.outer(norm_scaled_proj_V[i*200:(i+1)*200], norm_scaled_proj_T)
            dots = np.dot(scaled_proj_V[i*200:(i+1)*200], scaled_proj_T.T)
            similarities = dots / prods

            idx = similarities.argsort(axis=1)[:, ::-1][:, :50]
            ids = img_IDs[idx]

            for idsImg in ids:
                words = annotationsToWords(idsImg)
                tags.append([tag for tag, it in Counter(words).most_common(n_tags)])

    return tags
