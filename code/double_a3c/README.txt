for vanilla A3C ./train-atari.py --env Breakout-v0 --gpu 0
for double, less shared, no shared A3C first modify _get_NN_prediction, then run by
./double-A3C-atari.py --env Breakout-v0 --gpu 0