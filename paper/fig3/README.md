This directory runs the experiment corresponding to Figure 3 in the paper, and 
provides a plotting notebook, `plots.ipynb`, that can be run against the
`progress.csv` file generated by the authors in the `ray_results` directory, 
or against your own.

You might need to change the resource arguments in the call below to suit your
system but results should be qualitatively similar independent of the number
of CPUs and GPUs.

```bash
python -u train.py \
    --num-cpus 34 \
    --num-gpus 1 \
    --max-episode-steps 250
```

See the `examples/marl/rllib/README.md` for instructions on how to install
`rllib` in your environment, and other tips for running RLLib training.
