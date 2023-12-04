
# Shape Matching

This section of the repository contains implementations to the shape matching experiments .

## General Information

- **Source**: Portions of the code and data have been adapted and modified from Marin et al. [1]. Please refer to the comments within individual files for specific details regarding modifications, new additions, or unchanged elements.
- **Hardware Used**: Experiments were conducted using:

  - GeForce RTX 3090 with CUDA version 11.1.
  - Cluster environment with NVIDIA Tesla V100 GPUs.

## Setup: Environment and Data Preparation

1. **Installing PyTorch and Additional Requirements**:

   - Install PyTorch (version 1.9 was used for our experiments).

   - Install other necessary dependencies:
     ```shell
     pip install -r requirements.txt
     ```
3. **Data Acquisition**:

   - Download the dataset used in [1]:
     ```shell
     python ./data/download_data.py
     ```
   - Additionally, obtain "FAUST_noise_0.01.mat" from [this GitHub repository](https://github.com/riccardomarin/Diff-FMAPs-PyTorch/tree/main/data) and place it in the `./data/` directory.

## Training the Networks

- **Standard Training**:

  ```shell
  python ./code/train_basis.py --train_type 1 --lr 0.01 --pretrain 0 --name "ours"
  ```
- **Stochastic Training Variant**:

  ```shell
  python ./code/train_basis.py --number_extra_entries 1 --train_type 2 --lr 0.01 --pretrain 0 --name "stochastically"
  ```

  For additional settings and configurations, see `get_id.py`.

## Evaluation

- To evaluate the models, execute the `evaluation.ipynb` Jupyter Notebook. This notebook automates the processes of `test_faust.py` and `evaluation.m`.
- In the `evaluation.m` script, the geodesic distance matrix is set to use an approximate version by default, from [this GitHub repository](https://github.com/riccardomarin/Diff-FMAPs-PyTorch/tree/main/data). If you wish to utilize a geodesic distance matrix that you have computed yourself, navigate to lines 60 - 64 in the `evaluation.m` file and load your custom-calculated geodesic distance matrix.


> Reference:
> [1] Marin, R., Rakotosaona, M.J., Melzi, S., and Ovsjanikov, M. (2020). Correspondence learning via linearly-invariant embedding. In _Advances in Neural Information Processing Systems, 33_.
>
