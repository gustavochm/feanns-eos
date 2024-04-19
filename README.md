# Supporting Information: *"On the continuous modeling of fluid and solid states"*

This repository is part of the Supporting Information of the article *On the continuous modeling of fluid and solid states* by Gustavo Chaparro and Erich A. Müller. Preprint available [here](https://doi.org/10.26434/chemrxiv-2024-tjfj7). In this article, an equation of state based on artificial neural networks (FE-ANN(s) EoS) that continuously models fluid and solid states is presented. This EoS is showcased for the Mie particle.


$$ \mathcal{U}^{\textnormal{Mie}} = \epsilon  \underbrace{\left[ \frac{\lambda_r}{\lambda_r- \lambda_a} \left( \frac{\lambda_r}{\lambda_a}\right)^{\frac{\lambda_a}{\lambda_r - \lambda_a}} \right] }_{\mathcal{C}^{\textnormal{Mie}}} \left[ \left(\frac{\sigma}{r}\right)^{\lambda_r} -  \left(\frac{\sigma}{r}\right)^{\lambda_a} \right] $$

Here, $\mathcal{U}^{\textnormal{Mie}}$ refers to the interaction energy between two Mie particles, $\epsilon$ is the energy scale, $\sigma$ is the shape parameter, which is related to the particle's diameter and $r$ is the center-to-center distance. Finally, $\lambda_r$ and $\lambda_a$ are the repulsive and attractive exponents.


This repository includes the following information:
- [Databases of the Mie particle computed with molecular dynamics simulations](./database)
- [Python package to use the FE-ANN(s) EoS](./feanns_eos)
- [Parameters of the trained FE-ANN(s) EoS](./eos_params)
- Examples of how to use the FE-ANN(s) EoS. See Jupyter notebooks (1., 2., 3., and 4.)


------
### Prerequisites
- Numpy (tested on version 1.24.2)
- matplotlib (tested on version 3.6.3)
- pandas (tested on version 1.5.3)
- jax (tested on version 0.4.4)
- flax (tested on version 0.6.6)

-----
### FE-ANN(s) EoS 

The FE-ANN(s) EoS models the residual Helmholtz free energy as follows.

$$ A^{\textnormal{res}} = \textnormal{ANN}(\chi, \rho, 1/T) - \textnormal{ANN}(\chi, \rho=0, 1/T) $$

Here, $A^{\textnormal{res}}$ is the residual Helmholtz free energy, $\rho$ is the density, $T$ is the absolute temperature, and $\textnormal{ANN}$ refers to an artificial neural network. Finally, $\chi$ are the descriptors of the molecule (or pseudo-molecule) of interest, e.g., the molecular parameters of an interaction potential. In this work, the use of the FE-ANN(s) EoS is showcased for the Mie particle; in this case, the $\alpha_{vdw}$ is used as the molecular descriptor.

$$ \alpha_{\textnormal{vdw}} = \mathcal{C}^{\textnormal{Mie}} \left( \frac{1}{\lambda_a - 3} - \frac{1}{\lambda_r - 3}\right)  $$

The FE-ANN(s) EoS has been trained using first- and second-order derivative properties of the Mie particle. The following thermophysical properties are considered: compressibility factor, $Z$, second-virial coefficient, ${B}$, internal energy, ${U}$, isochoric heat capacity, ${C_V}$, thermal pressure coefficient, ${\gamma_V}$, isothermal compressibility, as ${\rho\kappa_T}$, thermal expansion coefficient, $\alpha_P$, adiabatic index, ${\gamma=C_P/C_V}$, and the Joule-Thomson coefficient, ${\mu_{JT}}$.

------
### License information

See ``LICENSE.md`` for information on the terms & conditions for usage of this software and a DISCLAIMER OF ALL WARRANTIES.

Although not required by the license, if it is convenient for you, please cite this if used in your work. Please also consider contributing any changes you make back, and benefit the community.
