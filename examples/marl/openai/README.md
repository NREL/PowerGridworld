The OpenAI code base has not been updated in recent years and
uses older versions of Python and other dependencies.

This document walks through a end-to-end installation of PowerGridworld and 
dependencies needed to run the MADDPG example.

To ensure that this example works, use an environment with Python 3.6.
We recommend using conda to install the base environment:

```
conda create -n maddpg python=3.6 -y
conda activate maddpg
```

Next clone and install PowerGridworld.

```
git clone https://github.com/NREL/PowerGridworld.git
cd PowerGridworld
pip install -e .
``` 

Next, we install our fork of MADDPG which is slightly modified from
the original implementation to remove an unneeded dependency on
the `multiagent` module.  We also install downgraded dependencies.

```
cd examples/marl/openai
pip install -r requirements.txt
git clone https://github.com/zxymark221/maddpg.git
cd maddpg
pip install -e .
```

Quick check of the installation:

```
python -c "import maddpg"
python -c "import gridworld"
```

Run the training script:

```
python train.py
```

After training, run `plot_learning_curves.py` plot the learning curves.