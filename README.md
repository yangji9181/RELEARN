RELEARN
---------------

Code for the paper "...." KDD 2019


Command
---------------

To train an RELEARN model with default setting, please see **run.sh**

Key Parameters
---------------

**sample_mode**: there are 4 type of loss, each one correspond to one component of sample mode. **n** is node feature reconstruct loss,  **l** is link prediction loss, **dc** is diffusion content reconstruction loss, **ds** is diffusion structure(link) prediction loss.

**diffusion_threshold**: filter out diffusion which contain nodes less than this threshold.

**neighbor_sample_size**: how many neighbor to aggregate in GCN layer.

**sample_size**: how many data to be used in one epoch for each sample mode. Note that for the two link prediction loss, sample size is the sum of positive sample size and negative sample size.

**negative_sample_size**: it is negative sample / positive sample.

**sample_embed**: the dimension of hidden state, also the dimension of learned embedding.

**relation**: number of relations to be used in variational inference.

**use_superv**: whether to add supervision in trainig.

**superv_ratio**: how many supervision to add, used in label efficiency experiments.

**a, b, c, d**: weights for different loss, main hyper-parameter to tune in practice.

