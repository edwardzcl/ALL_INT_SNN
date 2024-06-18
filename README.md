# ALL_INT_SNN: *An All Integer-based Spiking Neural Network with Dynamic  Threshold Adaptation*

***
**This code can be used as the supplemental material for the paper: "*An All Integer-based Spiking Neural Network with Dynamic Threshold Adaptation*". (Submitted to *Frontiers in Neuroscience*, June, 2024)** .
***

## Citation:
To be completed.

### **Features**:
- This supplemental material gives a reproduction function of ANN training, testing and converted SNN inference experiments in this paper.


## File overview:
- `README.md` - this readme file.<br>
- `MNIST` - the workspace folder for `LeNet` on MNIST.<br>
- `CIFAR10` - the workspace folder for VGG-Net on CIFAR10.<br>

## Requirements
### **Dependencies and Libraries**:
* python 3.5 (https://www.python.org/ or https://www.anaconda.com/)
* tensorflow_gpu 1.2.1 (https://github.com/tensorflow)
* tensorlayer 1.8.5 (https://github.com/tensorlayer)
* CPU: Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
* GPU: Tesla V100

### **Installation**:
To install requirements,

```setup
pip install -r requirements.txt
```
### **Datasets**:
* MNIST: [dataset](http://yann.lecun.com/exdb/mnist/), [preprocessing](https://github.com/tensorlayer/tensorlayer/blob/1.8.5/tensorlayer/files.py)
* CIFAR10: [dataset](https://www.cs.toronto.edu/~kriz/), 
[preprocessing](https://github.com/tensorlayer/tensorlayer/blob/1.8.5/tensorlayer/files.py)

## ANN Training
### **Before running**:
* Please installing the required package Tensorflow and Tensorlayer (using our modified version)
* Please note your default dataset folder will be `workspace/data`, such as `supplemental material/CIFAR10/data`

* Select the index of GPU in the training scripts (0 by default)

### **Run the code**:
for example (ANN training, *k=0*, VGG-Net, ternary weight, CIFAR10):
```sh
$ cd CIFAR10
$ python Quant_VGGNet_CIFAR10.py --k 0 --resume False --use_ternary True --learning_rate 0.001 --mode 'training'
```
## ANN Inference
### **Run the code**:
for example (ANN inference, *k=0*, VGG-Net, ternary weight, CIFAR10):
```sh
$ python Quant_VGGNet_CIFAR10.py --k 0 --resume True --use_ternary True --mode 'inference'
```
## SNN inference
### **Run the code**:
for example (SNN inference, *k=0*, *noise_ratio=0*, spiking VGG-Net, ternary weight, INT threshold, CIFAR10):
```sh
$ python Spiking_CNN1_CIFAR10.py --k 0 --use_ternary True --use_int True --noise_ratio 0
```
it will generate the corresponding log files including: `accuracy.txt`, `sop_num.txt`, `spike_collect.txt`, `spike_num.txt`, `abs_spike.txt`  and `sum_spike.txt` in `./figs/WQ_INT/k0`.

## Others
* We do not consider the synaptic operations in the input encoding layer and the spike output in the last classification layer (membrane potential accumulation ) for both original ANN counterparts and converted SNNs.<br>
* More instructions for running the code can be found from the arguments.

## Results
Our proposed methods achieve the following performances on MNIST, and CIFAR10 dataset:

### **MNIST**:
| Quantization Level  | Network Size  | Epochs | ANN | SNN | Time Steps |
| ------------------ |---------------- | -------------- | ------------- | ------------- | ------------- |
| Full-precision | 32C5-P2-64C5-P2-1000 |   200   |  99.45% | N/A | N/A |
| k=0 | 32C5-P2-64C5-P2-1000 |   200   |  99.45% | 99.45% |  4 |
||

### **CIFAR10**:
| Quantization Level  | Network Size  | Epochs | ANN | SNN | Time Steps |
| ------------------ |---------------- | -------------- | ------------- | ------------- | ------------- |
| Full-precision | 96C3-256C3-P2-384C3-P2-384C3-256C3-P2-1024-1024 | 200 | 93.32% | N/A | N/A |
| k=0 | 96C3-256C3-P2-384C3-P2-384C3-256C3-P2-1024-1024 | 200 | 93.32% | 93.15% |  8 |

## More question:<br>
- There might be a little difference of results for multiple training repetitions, because of the randomization. 
- Please feel free to reach out here or email: 2306394264@pku.edu.cn, if you have any questions or difficulties. I'm happy to help guide you.

