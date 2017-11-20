This repository includes python code and jupyter notebook for an introduction to Convolution Neural Network (CNN) and Recurrent Neural Network (RNN).


## Instruction

The notebooks have been tested using miniconda3 with python3.6 and python3.5.
Latest Keras (>=v2.0.9) is required to run the RNN code, which uses the `RNN` class.

To run the notebooks, here are some instructions
1. install the dependent libraries including, keras + tensorflow, jupyter notebook and others in requirements.txt
```   
    conda install tensorflow     
    # or with gpu 
    conda install tensorflow-gpu
    
    git clone https://github.com/fchollet/keras.git
    cd keras
    python setup.py install 
    
    conda install --file requirements.txt
```    
2. create the checkpoint folder, `ckpt` and `data/` folder.    
3. download the dataset into `data/` folder. Details are inside the notebooks.

Here are the [instructions](https://www.youtube.com/watch?v=8rjRfW4JM2I) to setup the GPU instance on Amazon AWS.
NOTE: select the nearest AWS region with GPU, e.g. Singapore. You can use cheaper GPU instances like g2.2xlarge. 
You can also use [Google Cloud](http://cs231n.github.io/gce-tutorial-gpus/), which has GPUs in Region Taiwan.


## Credit

We adapt code from [keras example](https://github.com/fchollet/keras/tree/master/examples) and [fast.ai courses](https://github.com/fastai/courses) to creat the notebooks.
* MNIST CNN, CNN and RNN for sentiment analysis, Char RNN, Seq2Seq are from keras.
* CNN fine-tune is from fast.ai.

