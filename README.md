# Deep Learning At Scale 

Welcome to the Deep Learning At Scale repository! This repository contains hands-on example code for the [O'Reilly book "Deep Learning At Scale"](https://www.oreilly.com/library/view/deep-learning-at/9781098145279/).

## Setup 

To set up your development environment, follow the instructions below.

### Python Environment 
To create your environment directly on the host machine, use the following instructions. These instructions are based on Python 3.9 but should work with higher versions of Python as well. 

1. To get python3.9 on Ubuntu, use:

    ```bash
    sudo apt-get update -yqq; 
    sudo apt-get install -y python3.9 python3.9-venv 
    ```

2. Create a Python virtual environment:
    ```bash
    python3.9 -m venv .venv
    source .venv/bin/activate
    ```

3. Install the dependencies using the following command:
    ```bash
    make install
    ```

4. If you need to update versions, use `lock` to update and freeze versions to the latest:
    ```bash
    make lock
    ```

5. To install a specific version of CUDA for PyTorch, use the following command. Override `--extra-index-url` as suited to your purpose:
    ```bash
    pip install deep-learning-at-scale --extra-index-url https://download.pytorch.org/whl/cu118
    ```

### Using Docker 
Alternatively, you can use Docker to run the hands-on exercises. To build your Docker container, use the following command:
```bash
docker build -t deep_learning_at_scale .
```

To run the Docker container in interactive mode, use the following command:
```bash
docker run -t -i -v $(PWD):/deep_learning_at_scale --name dls_handson --entrypoint bash deep-learning-at-scale
```

To run a specific example directly, use the following command:
```bash
docker run -t -i -v $(PWD):/deep_learning_at_scale --name dls_handson --entrypoint bash deep-learning-at-scale chapter_2 has_black_patch train
```

## Explicit Install of NVIDIA drivers and CUDA Runtime

If your host is equipped with an NVIDIA GPU, you will need to have the right version of NVIDIA drivers and corresponding OS kernel instructions installed. Refer to the [NVIDIA User guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for detailed instructions.

PyTorch provides a packaged installer with a few versions of CUDA install. However, for extensive debugging, profiling, or special CUDA-specific use cases, a full installation of NVIDIA Runtime may be required. Refer to the [CUDA Runtime installation guide](https://developer.nvidia.com/cuda-downloads) for instructions.

In some scenarios, the cupy package may also be needed. Refer to the [cupy installation guide](https://docs.cupy.dev/en/latest/install.html) for instructions.

### Configure different version of NVIDIA/CUDA toolkit
Once you have your NVIDIA driver, CUDA runtime, and CUPY kits installed, you can set the following three environment variables to point to the right location of the CUDA Toolkit:

1. Install CUDA 12.1 from [NVIDIA](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local):
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
   chmod +755 cuda_12.1.0_530.30.02_linux.run
   ```
2. Then install, using the following step:
   ```bash
   mkdir -p ~/cuda121
   DEBIAN_FRONTEND=noninteractive LC_ALL=C ./cuda_12.1.0_530.30.02_linux.run \
    --silent \
    --toolkit \
    --toolkitpath=~/cuda121
   ```
3. Then set these variables in your environment:
   ```bash
   export CUDA_HOME="/opt/cuda121/toolkit/"
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"${CUDA_HOME}/lib64"
   export PATH="${CUDA_HOME}/bin/":${PATH}
   ```

## Setting Up For Experiment tracking with AimHub
To track your experiments, you can use AimHub. Follow these steps:

1. Start the AIM server by running the following command:
    ```bash
    aim up
    ```
2. Access the web UI by visiting `http://127.0.0.1:43800` in your browser.
3. All training runs will be logged here, along with their corresponding exercise IDs.


## Using Tensorboard for monitoring and profiling

To monitor and profile your deep learning models, you can use Tensorboard. Follow these steps:

1. Start the Tensorboard UI by running the following command:
    ```bash
    tensorboard --logdir <path to logs created by exercise>
    ```

2. Access the Tensorboard web UI by visiting `http://localhost:6006/` in your browser.

Now you can easily monitor and analyze the performance of your models using Tensorboard.

