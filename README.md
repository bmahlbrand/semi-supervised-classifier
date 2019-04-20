Assuming PyTorch >= 1.0 and torchvision installed  

see [1](./hpc/README.md) for instructions to config / run on HPC  
see [2](./papers) for papers

TODO:
Compute stats of dataset
Whiten dataset on load
Implement dataset class and lay foundations for augmentation
training scaffolding
evaluation scaffolding
Dataset viz

JSON defined config for reproducible experiments (auto-write to experiment directory)

Modules:
ResNet
-Conv block
-Residual block
-Bottleneck block

Densenet 
-Dense block

Adversarial Encoder
Energy based GAN (probably useless)

configurable and swappable optimizers / schedulers 

Spectral norm
Optimal transport
Flow fields

GAN specific callbacks

double check dependencies (requirements.txt)
