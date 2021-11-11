# PowerGridworld:  A Framework for Multi-agent Reinforcement Learning in Power Systems

Authors:  David Biagioni, Xiangyu Zhang, Dylan Wald, Deepthi Vaidhynathan, 
Rhoit Chintala, Jennifer King, Ahmed S. Zamzam

All authors are with the [National Renewable Energy Laboratory (NREL)](https://www.nrel.gov).

### Basic installation instructions

Env setup:

```
conda create -n gridworld python=3.8 -y
conda activate gridworld

git clone git@github.com:NREL/PowerGridworld.git
cd PowerGridWorld
pip install -e .
pip install -r requirements.txt
```

Run the pytests to sanity check:

```
pytest tests/
pytests --nbmake examples/envs
```

### Examples

Example notebooks can be found here: `examples/envs`.

### Description

TODO:  Link to arxiv pre-print.

### Funding Acknowledgement

This work was authored by the National Renewable Energy Laboratory (NREL), 
operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of 
Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported by 
the Laboratory Directed Research and Development (LDRD) Program at NREL.
