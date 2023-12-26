<h1 align="center">
  <br>
Unveiling the effectiveness of the resurrecting RNNs
  <br>
</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/tal-rubinstein-131450a6
">Tal Rubinstein</a> 
  </p>
  <p align="center">
    <a href="https://www.linkedin.com/in/edo-cohen/?originalSubdomain=il">Supervised by Edo Cohen</a>
  </p>
<h4 align="center">Official repository of the project</h4>


> **Abstract:** Capturing long range dependencies is a fundamental challenge in many machine learning
tasks including natural language processing and time series analysis. In a recent series of
works, State Space Models (SSMs) have emerged and proven to be extremely effective in
modeling such dependencies, notably surpassing transformers in benchmarks such as Long
Range Arena (LRA). At their core, SSMs are recurrent neural networks (RNNs) with a
linear update to the hidden state, enabling efficient implementation and training on very
long sequences. Many variants of SSMs have been proposed with different architectural
designs. To this date, it is still unclear theoretically why SSMs are so effective. This
paper empirically investigates the effect of different design choices on the optimization and
generalization of SSMs. 


## Repository Organization

|File name         | Content |
|----------------------|------|
|`/configs/table2/mnist_guess_rnn.yaml`| Configurations file for the different experiments|
|`train_distributed_same_seeds.py`| Script to restore the results presented in the project's report|






