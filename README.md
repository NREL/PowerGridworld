# PowerGridworld:  A Framework for Multi-Agent Reinforcement Learning in Power Systems

Authors:  David Biagioni, Xiangyu Zhang, Dylan Wald, Deepthi Vaidhynathan, 
Rhoit Chintala, Jennifer King, Ahmed S. Zamzam

Corresponding author:  [David Biagioni](https://github.com/davebiagioni)

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

Examples of running various environments and MARL training algorithms can be found in [`examples`](./examples).

### Description

Please read our [preprint on arXiv](https://arxiv.org/abs/2111.05969) for 
more details.  

### Funding Acknowledgement

This work was authored by the National Renewable Energy Laboratory (NREL), 
operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of 
Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported by 
the Laboratory Directed Research and Development (LDRD) Program at NREL.

### Citation

If citing this work, please use the following:

```bibtex
@article{biagioni2021powergridworld,
  title={PowerGridworld: A Framework for Multi-Agent Reinforcement Learning in Power Systems},
  author={Biagioni, David and Zhang, Xiangyu and Wald, Dylan and Vaidhynathan, Deepthi, and Chintala, Rohit and King, Jennifer and Zamzam, Ahmed S.},
  journal={arXiv preprint arXiv:2111.05969},
  url={https://arxiv.org/abs/2111.05969},
  year={2021}
}
```

