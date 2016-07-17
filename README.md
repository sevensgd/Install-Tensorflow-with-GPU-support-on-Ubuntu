# Installing Tensorflow r0.8 on Ubuntu 14.04 with Anaconda

## Driver
- Software and Updates -> Additional Drivers -> pick NVIDIA 352.63 driver

## CUDA
- Download cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb from https://developer.nvidia.com/cuda-downloads
- Install CUDA
- sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
- sudo apt-get update
- sudo apt-get install cuda

## cuDNN
- Download cuDNN v4 Library for Linux from https://developer.nvidia.com/cudnn
- Install cuDNN
- extract the tgz archive
- sudo nautilius
- go /usr/local/cuda and copy the content of your cudnn folder there
- sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

# Configure CUDA
- add to your .bashrc:
- export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
- export CUDA_HOME=/usr/local/cuda

# Anaconda
- Download Anaconda from https://www.continuum.io/downloads
- bash Anaconda2-4.0.0-Linux-x86_64.sh
- conda create -n tensorflow python=2.7

# Enviroment
- To activate this environment, use:
- source activate tensorflow
- To deactivate this environment, use:
- $ source deactivate

# Tensorflow
- source activate tensorflow
- pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

# Test:
```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))
```

# I don't remember why I did that
- ipython-notebook from navigator
- conda install -c conda-forge ipywidgets

# Remote access
- https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh
- ssh -N -f -L localhost:8889:localhost:8888 remote_user@remote_host
- sudo apt-get install fail2ban
- sudo apt-get update
- sudo apt-get install openssh-server
- sudo ufw allow 22
