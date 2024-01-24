from tqdm import tqdm
from comparison.product_comparator import ProductComparator
from preprocessing.dataset_loader import PletsDataset
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import mean_squared_error


class HierachialMetric:
    """
        Custom metric for evaluating similarity scores on a multi-level hierarchy.
        Uses a combination of Kendall's Tau and Mean Squared Error between the achieved scored and the optimal ones.
        Requires a set of entity tuples with varying degrees of similarity, with the second entity being the most similar. 
    """
    def __init__(self) -> None:
        self.alpha = 0.5

    def evaluate_dataset(self, model: ProductComparator, test_data: PletsDataset, limit: int = 150) -> None:
        """
            Evaluate the model on a dataset with the custom metric.
            The dataset must be a PletsDataset instance, and the model must be a ProductComparator.
            The limit parameter specifies how many instances to evaluate.
        """
        count, score = 0, 0
        for batch in tqdm(test_data, total=limit):
            batch = np.array(batch).transpose()
            for instance in batch:
                similarities = [model.similarity(instance[0], instance[i+1]) for i in range(len(instance)-1)]
                count += 1
                score += self.evaluate_similarities(similarities)
                if count >= limit:
                    break
            if count >= limit:
                break
        return score/count

    def evaluate_similarities(self, similarities: np.array) -> float:
        """
            Calculate the metric score for a set of computed similarities.
            The similarities must be a numpy array of floats between 0 and 1, where 0 means no similarity and 1 means identical.
            The first element in the array is the most similar, and the last element is the least similar.
            Returns a float value between 0 and 1.
        """
        target = np.linspace(1, 0, len(similarities))
        target_order = np.linspace(1, len(similarities), len(similarities))
        similarity_order = len(similarities)-np.argsort(similarities).argsort()
        return (kendalltau(target_order, similarity_order).statistic+1)/2*self.alpha + (1-mean_squared_error(similarities, target))*(1-self.alpha)