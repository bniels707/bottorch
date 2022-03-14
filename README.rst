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

To make a prediction with a model::

  $ bottorch predict model.pth Icewave Chomp

To make a ranking of all competitors::

  $ bottorch rank model.pth

Limited "advanced" tuning parameters are available. `--l1` and `--l2` specify the sizes of the first two layers of the neural network. Their "ideal" size can be determined with hypertuning. Hypertuning also uses a configurable `--step_size` which determines the step size used for parameter optimization, larger step sizes are faster. There is also `--epochs` which can be used for number of tuning epochs run at each step. More epochs takes more time. For example::

  $ bottorch hypertune --step_size=1000 --epochs=5

Will print "best" l1 and l2 parameters, for example `Done! Best accuracy 58.7%, L1: 2032, L2: 1032`. To generate a model file using the parameters::

  $ bottorch tune --l1=2032 --l2=1032

The model will, by default, by saved to `model.pth`, which can be used for any future actions as well::

  $ bottorch predict model.pth Icewave Chomp
  $ bottorch rank model.pth
