## VTLN training

This directory contains the code for an experiment analyzing the impact of
speech duration used for VTLN training on the obtained VTLN warp coefficients.

It requires the Buckeye English dataset, which can be dowloaded from the link
provided
[here](https://github.com/bootphon/Zerospeech2015#zerospeech-challenge-2015).

To run the experiment, simply adapt the parameters in `run.sh` and launch it.

* It must be used with a cluster running SLURM (if not you must adapt the
  recipe).

* It assumes *shennong* and *ABXpy* are already installed in separated virtual
  environments. To install *ABXpy* in a dedicated conda environement just have a
  ``conda create --name abx -c coml abx``.

Once `run.sh` and all its jobs are terminated, run `./scripts/plot_warps.py` and
`./scripts/plot_abx.py` to generate the final plots in `data/plots`.
