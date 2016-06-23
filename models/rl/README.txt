Scripts for running DeepQLearing for the trading competition data.

Run the training:

python main_deepq.py

grep for Epoch to see the progress of the run...
Epoch 00000/99999 | Loss 1109.0891 | Win trades 0 | Total PnL -2420.7999999999956 
Epoch 00001/99999 | Loss 1091.1115 | Win trades 0 | Total PnL -2496.3999999999833 


iLoss is the sum of all the trainigs
Win Count: number of winning trades
Total reward: PnL 



Classfiles:

main_deepq.py       ... Main Program to start the run
Deep.py             ... Training the RL, hyperparameters (at the top)
Trading.py          ... rules for trading, P&L, this is the whole RL-environment
DataStore.py        ... reads and stores the training data from the trading competition. Change this for other input data 
ExperienceReplay.py ... stores known data to send it more than once through the net...
UFCNNModel.py       ... UFCNN or other neural network to approximate the Q function

testDataStore.py    ... Script to test the DataStore.py
Catch.py            ... old Catch game. Not used. Only for reference.

For a good intro into Deep Q Learning, see for instance:

https://www.nervanasys.com/demystifying-deep-reinforcement-learning/


