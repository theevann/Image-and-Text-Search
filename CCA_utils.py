# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as linalg


def mapVisualFeatures(V):
    return V


def mapTagFeatures(T):
    return T


def mapClassFeatures(C):
    return C


def computeCovMatrix(mappedFeatures):
    '''
    Takes the list of mappedFeatures (phi(X_i)) as input (two or
    three elements respectively for the two and three-view CCA)
    Returns:
    The covariance matrix S composed of all pairs of covariance
    matrices between the different views, and S_D, the bloc
    diagonal matrix composed of the self-covariance matrices
    for each view
    '''

    n_views = len(mappedFeatures)
    dimensions = np.zeros((1, n_views), dtype=np.int)
    dimensions[0] = [(mappedFeatures[i]).shape[1] for i in range(n_views)]

    dim = np.sum(dimensions)
    S = np.zeros((dim, dim))
    S_D = np.zeros((dim, dim))

    indices = np.cumsum(dimensions)
    indices = np.append(0, indices)

    for i in range(n_views):
        for j in range(i):
            S_ij = np.dot(mappedFeatures[i].T, mappedFeatures[j])
            S[indices[i]:indices[i+1], indices[j]:indices[j+1]] = S_ij

        S_ii = np.dot(mappedFeatures[i].T, mappedFeatures[i])
        S_D[indices[i]:indices[i+1], indices[i]:indices[i+1]] = S_ii

    S = S + S.T + S_D
    return S, S_D


def findMatrixAndEig(S, S_D, d, regularization, p):
    '''
    # inputs:
    S : the global covariance matrix between all pairs of "views"
    S_D : the bloc diagonal matrix composed of the self-covariance
    matrices for each view
    d : dimension of the common CCA space (number of kept
    eigenvectors and eigenvalues)
    regularization : added to the diagonal of the covariance matrix
    in order to regularize the problem
    p : the power to each the eigenvalues are elevated in the
    output D

    # outputs:
    W : the matrix composed of the d eigenvectors as columns
    D : diagonal matix given by the p-th power of the d
    corresponding eigenvalues
    '''

    ## REGULARIZE
    I_g = regularization * np.eye(len(S))
    #S = S + I_g
    S_D = S_D + I_g

    ## FIND EIGENVECTORS
    eigenValues, eigenVectors = linalg.eig(S, S_D)

    ## GET THE INDICES OF THE D LARGEST EIGENVALUES
    idx = eigenValues.argsort()[::-1][:d]

    ## BUILD W AND D
    D = np.diag(eigenValues[idx]**p)
    W = eigenVectors[:, idx]

    ## CHECK FOR CONSISTENCY
    if np.any(D.real != D):
        print('ERROR in CCA: Complex Eigenvalues!')
    D = D.real

    return W, D


def solveCCA(mappedFeatures, dimension, regularization, power):
    S, S_D = computeCovMatrix(mappedFeatures)
    W, D = findMatrixAndEig(S, S_D, dimension, regularization, power)
    return W, D
