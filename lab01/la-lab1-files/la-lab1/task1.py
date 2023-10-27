import numpy as np

A = np.random.rand(200, 10)

# -------------------- my answer to task1 ---------------------
B_answer = A - (A.sum(axis=0) / A.shape[0])  # answer
# np.mean()
print(f"B_answer is :\n{B_answer}")
