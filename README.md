# Kissing to Find a Match: Efficient Low-Rank Permutation Representation

## Abstract
Permutation matrices play a key role in matching and assignment problems across the fields, especially in computer vision and robotics. However, memory for explicitly representing permutation matrices grows quadratically with the size of the problem, prohibiting large problem instances. In this work, we propose to tackle the curse of dimensionality of large permutation matrices by approximating them using low-rank matrix factorization, followed by a nonlinearity. To this end, we rely on the Kissing number theory to infer the minimal rank required for representing a permutation matrix of a given size, which is significantly smaller than the problem size. This leads to a drastic reduction in computation and memory costs, e.g., up to $3$ orders of magnitude less memory for a problem of size $n=20000$, represented using $8.4×10^5$ elements in two small matrices instead of using a single huge matrix with $4×10^8$ elements. The proposed representation allows for accurate representations of large permutation matrices, which in turn enables handling large problems that would have been infeasible otherwise. We demonstrate the applicability and merits of the proposed approach through a series of experiments on a range of problems that involve predicting permutation matrices, from linear and quadratic assignment to shape matching problems.

## Code
These subfolders contain code to the following experiments:

- Point Cloud Alignment: *Pointclouds*
- Linear Assignment Problems and Quadratic Assignment Problems: *LAP and QAP*
- Shape Matching: *Shape-Matching*

Please open the README.md in the respective folders for information about the requirements, training and evaluation. We also provide information about the GPU and CUDA version, that were used for each experiment.
