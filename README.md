# Deep Belief Networks and Generative Modeling

Educational implementation of:
- Restricted Boltzmann Machines (RBM)
- Deep Belief Networks (DBN)
- Deep Neural Networks (DNN) with optional unsupervised pretraining
- Variational Autoencoders (VAEs)
- Generative Adversarial Networks (GANs)
- Denoising Diffusion Probabilistic Model (DDPM)
- Score-Based Generative Modeling (SGM)

The project is used to study generation (Binary AlphaDigits and MNIST) and classification (MNIST, binarized). 

## Project structure

```text
deep-belief-networks/
├── src/
│   ├── RBM.py
│   ├── DBN.py
│   ├── DNN.py
│   └── download_data.py
├── notebooks/
│   ├── DNN_MNIST.ipynb
│   ├── GANs_TP.ipynb
│   └── VAE.ipynb
├── binaryalphadigs.mat
└── Project description.pdf
```

## Requirements

- Python 3.9+
- `numpy`
- `matplotlib`
- `scikit-learn`
- `torch`
- `torchvision`
- Jupyter (for notebooks)

Install dependencies:

```bash
python -m pip install numpy matplotlib scikit-learn torch torchvision jupyter
```

## Data

- `binaryalphadigs.mat` is included in the repository root.
- MNIST is downloaded automatically by `torchvision.datasets.MNIST` when running the notebook.

Optional (download helper):

```bash
python src/download_data.py
```

## How to run

From the repository root:

```bash
jupyter notebook
```

Open `notebooks/DNN_MNIST.ipynb`.

If you get an import error (`No module named DNN`), add this at the top of the notebook:

```python
import os, sys
sys.path.append(os.path.abspath("../src"))
```

Then import with:

```python
from DNN import DNN
from DBN import DBN
from RBM import RBM
```

## Typical workflow (DNN on MNIST)

1. Prepare and binarize MNIST.
2. Initialize a DNN architecture (e.g., `[784, 200, 200]`).
3. (Optional) Pretrain hidden layers with RBMs (`pretrain_DNN`).
4. Train in supervised mode with backprop (`retropropagation`).
5. Evaluate error rate with `test_DNN`.
6. Compare softmax output probabilities on selected images.
7. Train the generative models each separately to see the generated images.

## Main API

- `RBM(p, q)`
	- `train_RBM(X, lr, batch_size, epoch)`
	- `generer_image_RBM(nb_images, iter_Gibbs, image_size)`

- `DBN()`
	- `init_DBN(layer_sizes)`
	- `train_DBN(X, iter, lr, batch_size)`
	- `generer_image_DBN(nb_images, iter_Gibbs)`

- `DNN()`
	- `init_DNN(layers_size, classes)`
	- `pretrain_DNN(data, iter, lr, batch_size)`
	- `retropropagation(iter, lr, batch_size, data, labels)`
	- `test_DNN(test_data, test_labels)`
	- `entree_sortie_reseau(data)`

## Notes

- This code is intended for teaching and experimentation.
- Training can be slow with large iteration counts; reduce dataset size/epochs for quick tests.
