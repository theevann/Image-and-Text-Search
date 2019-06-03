# -*- coding: utf-8 -*-
import torch
import scipy.linalg as linalg


class CCA():
    """
    dimension : dimension of the common CCA space (number of kept eigenvectors and eigenvalues)
    regularization : added to the diagonal of the covariance matrix in order to regularize the problem
    power : the power to each the eigenvalues are elevated in the output D
    """
    def __init__(self, dimension, regularization=1, power=-1):
        super(CCA, self).__init__()
        self.dimension = dimension
        self.regularization = regularization
        self.power = power

        self.features_list = []
        self.dims = []
        self.Ws = None
        self.D = None

    def solve(self, features_list):
        self.features_list = features_list
        self.dims = [f.size(1) for f in self.features_list]

        S, S_D = self.computeCovMatrix()
        self.findMatrixAndEig(S, S_D)

    def computeCovMatrix(self):
        '''
        # Inputs:
        features : takes the list of features (phi(X_i)) as input (two or
        three elements respectively for the two and three-view CCA)

        # Outputs:
        S : the covariance matrix composed of all pairs of covariance
        matrices between the different views,
        S_D : the bloc diagonal matrix composed of the self-covariance
        matrices for each view
        '''

        dims = torch.Tensor(self.dims)
        dim = torch.Tensor(dims).sum().int().item()
        S = torch.zeros((dim, dim))
        S_D = torch.zeros((dim, dim))

        indices = dims.cumsum(0)
        indices = torch.cat([torch.Tensor([0]), indices]).int()

        n_views = len(self.features_list)
        for i in range(n_views):
            for j in range(i):
                S_ij = self.features_list[i].t() @ self.features_list[j]
                S[indices[i]:indices[i+1], indices[j]:indices[j+1]] = S_ij
            S_ii = self.features_list[i].t() @ self.features_list[i]
            S_D[indices[i]:indices[i+1], indices[i]:indices[i+1]] = S_ii

        S = S + S.t() + S_D
        return S, S_D

    def findMatrixAndEig(self, S, S_D):
        '''
        # Inputs:
        S : the global covariance matrix between all pairs of "views"
        S_D : the bloc diagonal matrix composed of the self-covariance matrices for each view

        # Outputs:
        W : the matrix composed of the d eigenvectors as columns
        D : diagonal matix given by the p-th power of the d corresponding eigenvalues
        '''

        # REGULARIZE
        I_g = self.regularization * torch.eye(len(S))
        S_D = S_D + I_g

        # FIND EIGENVECTORS and GET THE INDICES OF THE D LARGEST EIGENVALUES
        eigenValues, eigenVectors = linalg.eig(S.numpy(), S_D.numpy())
        eigenValues, eigenVectors = torch.from_numpy(eigenValues.real), torch.from_numpy(eigenVectors.real)
        idx = eigenValues.argsort(descending=True)[:self.dimension]

        # BUILD W AND D
        self.D = torch.diag(eigenValues[idx] ** self.power)
        self.Ws = eigenVectors[:, idx].split(self.dims)

    def getSimilarities(self, feature_1, dim_1, dim_2):
        W_1 = self.Ws[dim_1]
        W_2 = self.Ws[dim_2]

        scaled_proj_1 = feature_1 @ W_1 @ self.D
        scaled_proj_2 = self.features_list[dim_2] @ W_2 @ self.D

        dots = scaled_proj_1 @ scaled_proj_2.t()
        prods = scaled_proj_1.norm() * scaled_proj_2.norm(dim=1)
        similarities = dots / prods

        return similarities.sort(descending=True)
