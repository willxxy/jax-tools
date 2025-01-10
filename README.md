# jax-tools
Tools I use for jax.
These are tested on Ubuntu 20.04.5 LTS (Focal Fossa).

All of these are based on nvcc --version == 11.8 and the following installation command:

`pip install "jax[cuda11_cudnn82]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

Run `install_check.py` to make sure jax is correctly installed with gpu. These are tested with A5000 and A6000 NVIDIA gpus.


[Good blog introducing jax through the lens of pytorch users](https://cloud.google.com/blog/products/ai-machine-learning/guide-to-jax-for-pytorch-developers)
