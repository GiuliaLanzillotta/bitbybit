# Bit by Bit 🧱 
**A shared research library for Continual Learning experiments.**

This repository contains the core utilities, network definitions, and environment wrappers shared across many research projects. By treating this code as a standalone library, improvements to the core infrastructure are immediately available to all dependent projects.

## 🚀 Installation 

Since this library is actively developed alongside specific experiments, it should be installed in Editable Mode. This allows you to modify the code in bitbybit and have those changes immediately reflected in your other project repositories without re-installing.


## 📦 Dependencies
The core library requires the following packages (automatically installed via pip install -e .):

    torch 
    torchvision
    avalanche-lib (CL baselines and metrics)
    wandb (Logging)
    numpy 
    pandas
    matplotlib 
    seaborn (Visualization)
    fvcore


## Setup Instructions for Collaborators
Clone this repository alongside your project repository.
Navigate to the bitbybit directory.
Install with pip in editable mode:

    git clone [https://github.com/GiuliaLanzillotta/bitbybit.git](https://github.com/GiuliaLanzillotta/bitbybit.git)'''

    cd bitbybit
    pip install -e .

What this does:
This creates a symbolic link to this folder in your Python environment. You can now use import bitbybit in any other script or notebook on your machine.


## 📂 Repository Structure
```environments.py```: Custom wrappers for Continual Learning benchmarks (Split-MNIST, CIFAR, etc.) and task generation logic.

```networks.py```: PyTorch definitions for shared model architectures (MLPs, ResNets, etc.) used across projects.

```utils.py```: General helper functions for training loops, logging, and configuration.

```viz_utils.py```: Shared plotting utilities for consistency across papers.

## 🛠 Usage Example
Once installed, you can import modules in your separate experiment repositories:

    import torch
    from bitbybit import environments, networks

The ```test_environments.ipynb``` notebook in bitbybit provides a simple example of how to use the environments module to create a Continual Learning environment.

###  Initialize a shared network architecture
    net_args = {
        "name": 'MLP' , # 'mlp', 'cnn', 'resnet50'
        "depth": 1
        "width": 256
        "activation": 'relu'  # 'relu', 'tanh', 'sigmoid'
    }
    net_args['num_classes_total'] = env.num_classes
    model = networks.get_network_from_name(config['name'], **config)

## 🤝 Contributing

If you find a bug while working on a specific project, fix it here in this repo. Push the changes to bitbybit separately from your experiment repo.*Warning: Remember that changes here affect all projects using this library. Ensure backwards compatibility when modifying core functions.*

Maintained by Giulia Lanzillotta