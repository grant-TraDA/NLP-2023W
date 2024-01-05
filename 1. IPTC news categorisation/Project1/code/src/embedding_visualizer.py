import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

class EmbeddingVisualizer:
    """Class for visualizing embeddings in 2D or 3D space with colored labels."""

    def __init__(self, embeddings, labels):
        """Initializes the EmbeddingVisualizer class.
        Args:
            embeddings: A list of embeddings.
            labels: A list of labels (categories).
        """
        self.embeddings = embeddings
        self.labels = labels

    def get_reduced_embeddings(self, dimension=3, method='pca'):
        """Returns the embeddings projected into lower dimension with a given method.
        Args:
            dimension: The dimension of the reduced embeddings.
            method: The method used for dimensionality reduction (PCA or t-SNE).
        Returns:
            The reduced embeddings.
        """
        if method == 'pca':
            return PCA(n_components=dimension).fit_transform(self.embeddings)
        elif method == 'tsne':
            return TSNE(n_components=dimension).fit_transform(self.embeddings)
        else:
            raise ValueError('method must be either pca or tsne')

    def visualize(self, dimension=3, method='pca', title=None, **kwargs):
        """Visualizes the embeddings in 2D or 3D space with colors indicating labels.
        Args:
            dimension: The dimension of the reduced embeddings.
            method: The method used for dimensionality reduction (PCA or t-SNE).
        """
        reduced_embeddings = self.get_reduced_embeddings(dimension, method)
        unique_labels = np.unique(self.labels)
        
        # Use a qualitative color map for better distinguishable colors
        # if small number of labels, use tab10, otherwise tab20
        if len(unique_labels) < 10:
            # custom tab10 with more distinguishable colors
            colors = np.array([[0.12156863, 0.46666667, 0.70588235],
                                 [1.        , 0.49803922, 0.05490196],
                                 [0.17254902, 0.62745098, 0.17254902],
                                 [0.83921569, 0.15294118, 0.15686275],
                                 [0.58039216, 0.40392157, 0.74117647],
                                 [0.54901961, 0.3372549 , 0.29411765],
                                 [0.89019608, 0.46666667, 0.76078431],
                                 [0.49803922, 0.49803922, 0.49803922],
                                 [0.7372549 , 0.74117647, 0.13333333],
                                 [0.09019608, 0.74509804, 0.81176471]])
            # colors to cmap
            colors = colors[:len(unique_labels)]
            colors = plt.cm.colors.ListedColormap(colors)

        else:
            colors = plt.cm.get_cmap('tab20', len(unique_labels))

        fig = plt.figure(figsize=(10, 10))
        
        if dimension == 2:
            for i, label in enumerate(unique_labels):
                indices = np.where(self.labels == label)
                plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=label, color=colors(i), **kwargs)
            # legend outside of plot
            plt.legend(bbox_to_anchor=(1.1, 1.05))
            plt.title(title)

        elif dimension == 3:
            ax = fig.add_subplot(111, projection='3d')
            for i, label in enumerate(unique_labels):
                indices = np.where(self.labels == label)
                ax.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], reduced_embeddings[indices, 2], label=label, color=colors(i), **kwargs)
            ax.set_title(title)
            # legend outside of plot
            ax.legend(bbox_to_anchor=(1.1, 1.05))
        plt.show()
