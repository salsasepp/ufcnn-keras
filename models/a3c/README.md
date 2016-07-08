# A3C Trading

Integration of our Trading Framework into [miyosuda's A3C code] (https://github.com/miyosuda/async_deep_reinforce) based on the paper "Asynchronous Methods for Deep Reinforcement Learning" , published in [Arxiv] (https://arxiv.org/abs/1602.01783) bei V. Mnih et al.. The implemented algorithms are A3C and A3C LSTM.

The code runs technically, but it did not produce good results for a large Sine curve (Sine_Long.csv). But may be I did not run it long enough....

The File constants.py contains important constants, e.g. use LSTM or not, number of threads, use GPU,.... and needs to be modified.

The jobs run forever. Only if you stop a job via <CTRL> C (interactive job) or `kill -INT processid` (Job in the background), the program writes data in the checkpoints/ dir that contains all necessary data to restart. Please note that if you change a network parameter in constants.py, you must remove or rename the checkpoints/ directory, otherwise a3c.py might terminate with an error message.


Sine_long.csv must be renamed into prod_data_20130103v.txt and placed in the training_data_large/ directory

To start the optimization, run `python a3c.py`


To view the results of an optimisation on the testing days, run `python Tradingresults.py`. You will need a checkpoint first, and the testing days need to be set in constants.py

## System Requirements

- pythpon 2.7 or 3.x
- TensorFlow
- numpy, pandas, ...

## TODO

- Write code to analyse the network and run it over unseen data
- Remove the 84x84 network size constraint
- Make the network use 1D Convolition instead of 2D
- Integrate other network layers...

