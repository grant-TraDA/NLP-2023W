import time
from comparison.product_comparator import ProductComparator
from .table import print_table

def item_comparison(model: ProductComparator, text1: str, text2: str) -> None:
    """
        Prints a prettified table of the similarity between two items.
        Example output for 'red apple' and 'green apple':

        ######## Comparison ########
        # Red apple                #
        # Green apple              #
        ############################
        # Similarity: 0.99230      #
        # Execution time: 122.94ms #
        ############################
    """

    start_time = time.time()
    for _ in range(100):
        similarity = model.similarity_raw(text1, text2)
    end_time = time.time()
    execution_time = (end_time - start_time) * 10

    header = " Comparison "
    item1 = f" {text1} "
    item2 = f" {text2} "
    similarity = f" Similarity: {similarity:.5f} "
    execution_time = f" Execution time: {execution_time:.2f}ms "
    print_table(header, [item1, item2, "", similarity, execution_time, ""])
    print("")