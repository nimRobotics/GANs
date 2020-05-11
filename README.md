# GANs

This repo contains the implementations for [GAN](https://arxiv.org/abs/1406.2661), [WGAN](https://arxiv.org/abs/1701.07875), and [DCGAN](https://arxiv.org/abs/1511.06434). This was done as a course project and the slides are available [here](https://nimrobotics.github.io/assets/projects/gan.pdf).

**Author(s):** Aakash Yadav

### Usage
To run 
  - GAN with MNIST: `python3 gan.py`
  - WGAN with MNIST: `python3 wgan.py`
  - DC GAN with custom face dataset: `python3 dcgan.py --dataset folder --cuda --dataroot faces_dir --niter 300 --outf output_dir`

To plot the metrics
```
import load_met
metrics,d_losses,g_losses=load_met.load_data("outputs_wgan")
import plot_met
plot_met.plot_data(metrics,d_losses,g_losses)
``` 
 
### Dataset
  - [MNIST](http://yann.lecun.com/exdb/mnist/)
  - Custom Face Dataset (Not available publicly)

### Directory structure
	.
	+-- LICENSE
	+-- README.md
	+-- requirements.txt
	+-- face_dir
	|   +-- train
	|   +-- test
	+-- gan.py
	+-- wgan.py
	+-- dcgan.py
	+-- create_dataset.py
	+-- metric.py
	+-- load_met.py
	+-- plot_met.py
	

### Requirements
  - Python 3.6.9
  - Other dependencies can be installed using `pip3 install -r requirements.txt`
  
### TODO
  - [x] ReadMe
  - [ ] Refactor
  - [ ] Jupyter Notebook
