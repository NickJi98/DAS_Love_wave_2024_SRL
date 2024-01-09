## This repository corresponds to the following paper:

Ji, Q., B. Luo, and B. Biondi (2024). Exploiting the Potential of Urban DAS Grids: Ambient-Noise Subsurface Imaging Using Joint Rayleigh and Love Waves, *Seismol. Res. Lett.* XX, 1–18, doi: 10.1785/0220230104

## Contents

* `main.ipynb` &ensp; Main Jupyter Notebook to reproduce figures in the paper.
* `my_func/` &ensp; Python functions called in the notebook.
* `channel_info/` &ensp; Channel locations and channel pair information.
* `disp_maps/` &ensp; Pre-computed dispersion maps. Can be reproduced by the notebook.
* `noise_cc/` &ensp; MATLAB codes for synthetic DAS ambient noise cross-correlation (`fiber_profile.m`).

## Usage

The cross-correlation data can be downloaded from: http://dx.doi.org/10.5281/zenodo.7761930

Code blocks in `main.ipynb` include comments describing which figures to reproduce. For details of our analysis, please check the source codes under `my_func/`.

## Reference

This project uses the following Python package (version 2.0.1):

* Luu, K. (2021). evodcinv: Inversion of dispersion curves using evolutionary algorithms, doi: 10.5281/zenodo.5785565.

Github page: https://github.com/keurfonluu/evodcinv/tree/v2.0.1

## Citation

If you may find our work helpful in your publications, please consider citing our paper.

```
@article{Ji_2024_SRL,
  title = {Exploiting the {{Potential}} of {{Urban DAS Grids}}: {{Ambient-Noise Subsurface Imaging Using Joint Rayleigh}} and {{Love Waves}}},
  author = {Ji, Qing and Luo, Bin and Biondi, Biondo},
  year = {2024},
  journal = {Seismological Research Letters},
  doi = {10.1785/0220230104},
}
```

## License
This repository is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode).


