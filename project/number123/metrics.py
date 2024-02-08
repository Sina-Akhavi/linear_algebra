import numpy as np

def clustering_score(true_labels, predicted_labels):
    """
    Calculate the clustering score to assess the accuracy of predicted labels compared to true labels.

    Parameters:
    - true_labels: List or numpy array, true cluster labels for each data point.
    - predicted_labels: List or numpy array, predicted cluster labels for each data point.

    Returns:
    - score: float, clustering score indicating the accuracy of predicted labels.
    """

    # TODO: Calculate and return clustering score
    TP, TN, FP, FN = 0, 0, 0, 0
    
    n = len(true_labels)
    for i in range(n - 1):
        for j in range(i + 1, n):

            if predicted_labels[i] == predicted_labels[j]: # positive pair
                if true_labels[i] == true_labels[j]: # TP
                    TP = TP + 1
                else: # FP
                    FP = FP + 1
            else: # negative pair
                if true_labels[i] != true_labels[j]: # TN
                    TN = TN + 1
                else: # FN
                    FN = FN + 1
    

    return (TN + TP) / (TN + TP + FN + FP)


# res = [0, 0, 1, 2, 1]
# gt = [2, 2, 3, 1, 3]

# ri_output = clustering_score(gt, res)

# print("ri_output=", ri_output)



    
