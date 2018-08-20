import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PCA:
    def __init__(self,data):
        self.data = data
        self.X = np.array(data)  # row vector
        self.N = self.X.shape[0]  # shape (row, column)
        self.dim = self.X.shape[1]  # shape (row, column)
        self.mean = np.sum(self.X, axis=0)/self.N  # compute sum of each column
        self.cov = None

        self.show("X(n*m)",self.X)
        self.show("N",self.N)
        self.show("dim",self.dim)
        self.show("mean",self.mean)

    def show(self,title,value):
        split_line = "-----------"
        print(title+"\n",value,"\n",split_line)

    def normalize(self):
        self.X = self.X - self.mean.transpose()
        self.show("normalize X",self.X)

    def covariance(self):
        self.cov = self.X.transpose().dot(self.X)
        self.show("covariance",self.cov)

    def eigen_decomposition(self):
        eigenvalues, eigenvectors = eig(self.cov)  # column vector is eigen vector
        self.show("eigenvalues", eigenvalues)
        self.show("eigenvectors(m*k)", eigenvectors)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    def get_value(self,i):
        return self.eigenvalues[i]

    def get_vector(self,i):
        return self.eigenvectors[:,i]

    def verify(self):
        reconstruct_cov = np.round(self.eigenvectors.dot(np.diag(self.eigenvalues)).dot(np.linalg.inv(self.eigenvectors)))
        self.show("reconstruct cov",reconstruct_cov)

        cov_eigenvectors = np.round(self.cov.dot(self.eigenvectors))
        self.show("cov*eigenvectors", cov_eigenvectors)
        eigenvectors_eigenvalues = np.round(self.eigenvectors.dot(np.diag(self.eigenvalues)))
        self.show("eigenvectors*eigenvalues", eigenvectors_eigenvalues)
        self.show("equal(all entry should be true)", cov_eigenvectors == eigenvectors_eigenvalues)

    def run(self):
        self.normalize()
        self.covariance()
        self.eigen_decomposition()

    def project(self,k):
        new_X = self.X.dot(self.eigenvectors)
        new_X = new_X[:,k]
        self.show("new_X(n*k)", new_X)
        return new_X

    def visualize(self):
        v0 = np.array(pca.get_vector(0))
        v1 = pca.get_vector(1)
        mean = pca.mean

        plt.arrow(mean[0], mean[1], -v0[0] * 3, -v0[1] * 3, fc='r', ec='r')
        plt.arrow(mean[0], mean[1], v0[0] * 2, v0[1] * 2, head_width=0.3, head_length=0.2, fc='r', ec='r')
        plt.arrow(mean[0], mean[1], -v1[0] * 5, -v1[1] * 5, fc='b', ec='b')
        plt.arrow(mean[0], mean[1], v1[0] * 8, v1[1] * 8, head_width=0.3, head_length=0.2, fc='b', ec='b')

        self.show("v0", v0)
        self.show("v1", v1)
        self.show("mean", mean)

        plt.xlim(0,10)
        plt.ylim(-1,13)

        plt.scatter(self.data[:, 0], self.data[:, 1], color="black")
        color = ["red","blue"]
        for k in range(self.dim):
            v = pca.get_vector(k)
            scale = np.array(pca.project(k))
            proj = np.matrix(scale).transpose().dot(np.matrix(v))
            self.show("proj",proj)
            projpoints = np.matrix(proj)
            for i in range(proj.shape[0]):
                projpoints[i, :] = np.add(projpoints[i, :],mean)
            plt.scatter(np.array(projpoints[:, 0]),np.array(projpoints[:, 1]), color=color[k],marker='x')

        plt.show()

if __name__ == '__main__':
    np.random.seed(10)
    # data = np.array([[1,2,7],
    #                 [4,5,6],
    #                 [1,5,8]])  # row vector
    data = np.random.multivariate_normal([5,7],[[1,0],[0,10]],20)
    pca = PCA(data)
    pca.run()
    pca.visualize()
