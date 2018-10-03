# manifold-cnn
PyTorch implementation of manifold valued convolutions - developed in CVGMI lab at the University of Florida.
Part of the code used for experiments in the paper ManifoldNet: A Deep Network Framework for Manifold-valued Data.

### Paper Authors: Rudrasis Chakraborty, Jose Bouza, Baba C. Vemuri
### Code Author: Jose Bouza

This repository currently includes PyTorch implementations of the Grassmann averaging block, a dimensionality reduction layer
that can be added to an autoencoder to reduce the dimensionality of the latent space.

To run the network on an example video, make sure you have the required dependencies and use the following command in the hd_vid folder:
```python
python3 model.py
```

Note that training is memory intensive for high resolution videos like the one included with the code. The original system the network was trained on had 64 GB RAM.

## Dependencies 
```
 PyTorch v0.4
 OpenCV
```
