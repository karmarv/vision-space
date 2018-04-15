------------------------------------------------------------
- Experiment with using neural networks for a recognition task
- Datasets:
-   Mnist http://yann.lecun.com/exdb/mnist
-   CiFar10 https://www.cs.toronto.edu/~kriz/cifar.html
------------------------------------------------------------

# Demonstrate components 
1. process images (input, batch, format conversion, etc.)
2. perform training (with proper batching and validating)
3. estimate performance (e.g., error rate vs. run time and iterations, resource usage statistics)

### Note: Use CNN with a choice of number of blocks, number of layers, number of neurons per layer etc

- Exec  
```
    $ python cnn_recog_mnist.py
```
- Program Dumps the results in stdio/console
- plots are dumped in output folder
