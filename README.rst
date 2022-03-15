Setup
-----

It is recommended to develop using a `pipenv <https://pipenv.pypa.io/en/latest/>`_.

A development environment can be set up automatically::

  $ pipenv install --dev --pre

You can work directly inside the development environment::

  $ pipenv shell

Use
---

Save `BotTorch_Data-S6-Competitors.csv` and `BotTorch_Data-Training.csv` to your working directory.

To generate a model file::

  $ bottorch tune

This will generate a `model.pth` file.

To make a prediction after a model has been created (used `model.pth` by default)::

  $ bottorch predict Icewave Chomp

To make a ranking of all competitors, assuming competitors are saved in `competitors.csv` with one competitor name per line::

  $ bottorch rank --competitors competitors.csv

Assuming the competitors file is in order of first seed to last seed, you can simulate a single elimination bracket::

  $ bottorch bracket --competitors competitors.csv

Limited "advanced" tuning parameters are available. `--l1` and `--l2` specify the sizes of the first two layers of the neural network. Their "ideal" size can be determined with hypertuning. Hypertuning also uses a configurable `--step_size` which determines the step size used for parameter optimization, larger step sizes are faster. There is also `--epochs` which can be used for number of tuning epochs run at each step. More epochs takes more time. For example::

  $ bottorch hypertune --step_size=1000 --epochs=5

Will print "best" l1 and l2 parameters, for example `Done! Best accuracy 58.7%, L1: 2032, L2: 1032`. To generate a model file (`model.pth`) using the parameters::

  $ bottorch tune --l1=2032 --l2=1032
