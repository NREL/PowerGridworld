# PowerGridworld:  A Framework for Multi-Agent Reinforcement Learning in Power Systems

![ci_workflow](https://github.com/NREL/PowerGridworld/actions/workflows/main.yml/badge.svg)
![codeql_workflow](https://github.com/NREL/PowerGridworld/actions/workflows/codeql-analysis.yml/badge.svg)

Authors:  David Biagioni, Xiangyu Zhang, Dylan Wald, Deepthi Vaidhynathan, 
Rhoit Chintala, Jennifer King, Ahmed S. Zamzam

Corresponding author:  [David Biagioni](https://github.com/davebiagioni)

All authors are with the [National Renewable Energy Laboratory (NREL)](https://www.nrel.gov).

### Description

PowerGridworld provides users with a lightweight, modular, and customizable
framework for creating power-systems-focused, multi-agent Gym
environments that readily integrate with existing training frameworks for reinforcement learning (RL). Although many frameworks exist for training multi-agent RL (MARL) policies, none can rapidly prototype and develop the environments themselves,
especially in the context of heterogeneous (composite, multidevice) power systems where power flow solutions are required to
define grid-level variables and costs. PowerGridworld is an opensource software package that helps to fill this gap. To highlight
PowerGridworld’s key features, we include two case studies
and demonstrate learning MARL policies using both OpenAI’s
multi-agent deep deterministic policy gradient (MADDPG) and
RLLib’s proximal policy optimization (PPO) algorithms. In both
cases, at least some subset of agents incorporates elements of the
power flow solution at each time step as part of their reward
(negative cost) structures.

Please refer to our [published paper](https://dl.acm.org/doi/abs/10.1145/3538637.3539616) or [preprint on arXiv](https://arxiv.org/abs/2111.05969) for 
more details.  Data and run scripts used to generate figures in the paper
are available in the [`paper`](./paper) directory.

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

We have also added a `pyproject.toml` file to support the use of [poetry](https://python-poetry.org/docs/).  If using poetry, simply do `poetry install`.

Run the pytests to sanity check:

```
pytest tests/
pytests --nbmake examples/envs
```

### Examples

Examples of running various environments and MARL training algorithms can be found in [`examples`](./examples).


### Funding Acknowledgement

This work was authored by the National Renewable Energy Laboratory (NREL), 
operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of 
Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported by 
the Laboratory Directed Research and Development (LDRD) Program at NREL.

### Citation

If citing this work, please use the following:

```bibtex

@inproceedings{biagioni2021powergridworld,
  author = {Biagioni, David and Zhang, Xiangyu and Wald, Dylan and Vaidhynathan, Deepthi and Chintala, Rohit and King, Jennifer and Zamzam, Ahmed S.},
  title = {PowerGridworld: A Framework for Multi-Agent Reinforcement Learning in Power Systems},
  year = {2022},
  isbn = {9781450393973},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3538637.3539616},
  doi = {10.1145/3538637.3539616},
  booktitle = {Proceedings of the Thirteenth ACM International Conference on Future Energy Systems},
  pages = {565–570},
  numpages = {6},
  keywords = {deep learning, power systems, OpenAI gym, reinforcement learning, multi-agent systems},
  location = {Virtual Event},
  series = {e-Energy '22}
}
```

