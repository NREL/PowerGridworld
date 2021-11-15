The OpenAI code base has not been updated in recent years and
uses older versions of Python and other dependencies.

__There is a security vulnerability associated with the tensorflow version
used in this example.  [See this pull request for details](https://github.com/NREL/PowerGridworld/pull/10) 
-- run at your own risk!__

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

Update your environment for this example.  To do this, you must install
the following packages (e.g., using `pip` directly, or by creating the
appropriate `requirements.txt` file -- but see warning above!).

```
opendssdirect.py
gym==0.18.3
tensorflow==1.8.0
pandas==1.1.5
matplotlib==3.3.4
```

Next, we install our fork of MADDPG which is slightly modified from
the original implementation to remove an unneeded dependency on
the `multiagent` module.  We also install downgraded dependencies.

```
cd examples/marl/openai
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
