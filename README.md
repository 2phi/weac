<!-- LOGO AND TITLE-->
<p align="right"><img src="https://github.com/2phi/weac/raw/main/img/logo.png" alt="Logo" width="80" height="80"></p>

# WEAC &nbsp;·&nbsp; Weak Layer Anticrack Nucleation Model

<!-- BADGES -->
<!-- [![Weac][weac-badge]][weac-url] -->
<!-- [![Python][python-dist-badge]][pypi-url] -->
<!-- [![Downloads][pypi-downloads-badge]][pypi-url] -->
<!-- [![Stargazers][stars-badge]][stars-url] -->
<!-- [![Contributors][contributors-badge]][contributors-url] -->
<!-- [![Issues][issues-badge]][issues-url] -->
<!-- [![Forks][forks-badge]][forks-url] -->
<!-- [![DOI](https://zenodo.org/badge/203163531.svg)](https://zenodo.org/badge/latestdoi/203163531) -->
[![Release][release-badge]][release-url]
[![PyPI][pypi-badge]][pypi-url]
[![DOI][doi-badge]][doi-url]\
Implementation of closed-form analytical models for the analysis of dry-snow slab avalanche release.

[View the demo](https://github.com/2phi/weac/blob/main/demo/demo.ipynb) · 
[Report a bug](https://github.com/2phi/weac/issues) · 
[Request a feature](https://github.com/2phi/weac/issues) · 
[Read the docs](https://2phi.github.io/weac/) · 
[Cite the software](https://github.com/2phi/weac/blob/main/CITATION.cff)

<!-- TABLE OF CONTENTS -->
## Contents
1. [About the project](#about-the-project)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Roadmap](#roadmap)
5. [Release history](#release-history)
6. [How to contribute](#how-to-contribute)
7. [License](#license)
8. [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About the project

WEAC implements closed-form analytical models for the [mechanical analysis of dry-snow slabs on compliant weak layers](https://doi.org/10.5194/tc-14-115-2020), the [prediction of anticrack onset](https://doi.org/10.5194/tc-14-131-2020), and, in particular, allows for the analysis of stratified snow covers. The model covers propagation saw tests (a), and uncracked (b) or cracked (c) skier-loaded buried weak layers.

<img src="https://github.com/2phi/weac/raw/main/img/bc.png" alt="Boundary conditions" width="500"/>

Please refer to the companion papers for model derivations, illustrations, dimensions, material properties, and kinematics:

- Rosendahl, P. L., & Weißgraeber, P. (2020). Modeling snow slab avalanches caused by weak-layer failure – Part 1: Slabs on compliant and collapsible weak layers. The Cryosphere, 14(1), 115–130. https://doi.org/10.5194/tc-14-115-2020
- Rosendahl, P. L., & Weißgraeber, P. (2020). Modeling snow slab avalanches caused by weak-layer failure – Part 2: Coupled mixed-mode criterion for skier-triggered anticracks. The Cryosphere, 14(1), 131–145. https://doi.org/10.5194/tc-14-131-2020

Written in [🐍 Python](https://www.python.org)  and built with [<span style="color:#498B60">⚛</span> Atom](https://atom.io), [🐙 GitKraken](https://www.gitkraken.com), and [🪐 Jupyter](https://jupyter.org). Note that [release v1.0](https://github.com/2phi/weac/releases/tag/v1.0.0) was written and built in [🌋 MATLAB](https://www.mathworks.com/products/matlab.html). 

<!-- 
[![Python](https://img.shields.io/badge/Python-306998.svg?style=flat&logo=python&logoColor=white&label&labelColor=gray)](https://www.python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-e67124.svg?style=flat&logo=jupyter&logoColor=white&label&labelColor=gray)](https://jupyter.org)
[![Atom](https://img.shields.io/badge/Atom-498b60.svg?style=flat&logo=atom&logoColor=white&label&labelColor=gray)](https://atom.io)
[![GitKraken](https://img.shields.io/badge/GitKraken-179287.svg?style=flat&logo=gitkraken&logoColor=white&label&labelColor=gray)](https://www.gitkraken.com) -->



<!-- INSTALLATION -->
## Installation

Install globally using the `pip` Package Installer for Python
```sh
pip install -U weac
```
or clone the repo
```sh
git clone https://github.com/2phi/weac
```
for local use.

Needs
- [Numpy](https://numpy.org/) for matrix operations
- [Scipy](https://www.scipy.org/) for solving optimization problems
- [Pandas](https://pandas.pydata.org/) for data handling
- [Matplotlib](https://matplotlib.org/) for plotting

<!-- USAGE EXAMPLES -->
## Usage

The following describes the basic usage of WEAC. Please refer to the [demo](https://github.com/2phi/weac/blob/main/demo/demo.ipynb) for more examples and read the [documentation](https://2phi.github.io/weac/) for details.

Load the module.
```python
import weac
```
Choose a snow profile from the database (see [demo](https://github.com/2phi/weac/blob/main/demo/demo.ipynb)) or create your own as a 2D array where the columns are density (kg/m^2) and layer thickness (mm). One row corresponds to one layer counted from top (below surface) to bottom (above weak layer). 
```python
myprofile = [[170, 100],  # (1) surface layer
             [190,  40],  # (2)
             [230, 130],  #  :
             [250,  20],  #  :
             [210,  70],  # (k)
             [380,  20],  #  :
             [280, 100]]  # (L) last slab layer above weak layer
```
Create a model instance with optional custom layering.
```python
skier = weac.Layered(system='skier', layers=myprofile)
```
Calculate lists of segment lengths, locations of foundations, and position and magnitude of skier loads from the inputs total length `L` (mm), crack length `a` (mm), and skier weight `m` (kg). We can choose to analyze the situtation before a crack appears even if a crack length > 0 is set by replacing the `'crack'` key thorugh the `'uncracked'` key.
```python
segments = skier.calc_segments(L=10000, a=300, m=80)['crack']
```
Assemble the system of linear equations and solve the boundary-value problem for the free constants `C` providing the inclination `phi` (counterclockwise positive) in degrees.
```python
C = skier.assemble_and_solve(phi=38, **segments)
```
Prepare the output by rasterizing the solution vector at all horizontal positions `xq`. The result is returned in the form of the ndarray `zq`. We also get `xb` that only contains x-coordinates that lie on a foundation.
```python
xq, zq, xb = skier.rasterize_solution(C=C, phi=38, **segments)
```
Visualize the results.
```python
# Visualize deformations as a contour plot
weac.plot.contours(skier, x=xq, z=zq, window=200, scale=100)

# Plot slab displacements (using x-coordinates of all segments, xq)
weac.plot.displacements(skier, x=xq, z=zq, **segments)

# Plot weak-layer stresses (using only x-coordinates of bedded segments, xb)
weac.plot.stresses(skier, x=xb, z=zq, **segments)
```
Compute output quantities for exporting or plotting.
```python
# Slab deflections (using x-coordinates of all segments, xq)
x_cm, w_um = skier.get_slab_deflection(x=xq, z=zq, unit='um')

# Weak-layer shear stress (using only x-coordinates of bedded segments, xb)
x_cm, tau_kPa = skier.get_weaklayer_shearstress(x=xb, z=zq, unit='kPa')
```

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/2phi/weac/issues) for a list of proposed features and known issues.

### v2.3
- [ ] Finite fracture mechanics implementation for layered snow covers

### v2.2
- [x] Make sign of inclination `phi` consistent with the coordinate system (positive counterclockwise)
- [x] Add dimension arguments to field-quantity methods
- [x] Improved aspect ratio of profile views and contour plots
- [x] Improved plot labels
- [x] Provide convenience methods for the export of weak-layer stresses and slab deformations
- [ ] Add demo dif

## Release history

### v2.1
- Consistent use of coordinate system with downward pointing z-axis
- Consitent top-to-bottom numbering of slab layers
- Implementation of PSTs cut from either left or right side

### v2.0
- Completely rewritten in 🐍 Python
- Coupled bending-extension ODE solver implemented
- Stress analysis of arbitrarily layered snow slabs
- FEM validation of
  - displacements
  - weak-layer stresses
  - energy release rates in weak layers
- Documentation
- Demo and examples

### v1.0

- Written in 🌋 MATLAB
- Deformation analysis of homogeneous snow labs
- Weak-layer stress prediction
- Energy release rates of cracks in weak layers
- Finite fracture mechanics implementation
- Prediction of anticrack nucleation


<!-- CONTRIBUTING -->
## How to contribute

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazingfeature`)
3. Commit your changes (`git commit -m 'Add some amazingfeature'`)
4. Push to the branch (`git push origin feature/amazingfeature`)
5. Open a pull request


<!-- LICENSE -->
## License

Copyright 2phi GbR, 2021.

We currently do not offer an open source license. Please contact us for private licensing options.

<!-- Do not forget to pick a license provide details in the `LICENSE` file. -->


<!-- CONTACT -->
## Contact

E-mail: mail@2phi.de · Web: https://2phi.de · Project Link: [https://github.com/2phi/weac](https://github.com/2phi/weac) · Project DOI: [http://dx.doi.org/10.5281/zenodo.5773113](http://dx.doi.org/10.5281/zenodo.5773113)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-badge]: https://img.shields.io/github/contributors/2phi/weac.svg?style=flat&logo=github&color=yellow

[forks-badge]: https://img.shields.io/github/forks/2phi/weac.svg?&color=blueviolet&style=flat&logo=github

[stars-badge]: https://img.shields.io/github/stars/2phi/weac.svg?style=flat&logo=github&color=orange

[issues-badge]: https://img.shields.io/github/issues/2phi/weac.svg?style=flat&logo=github

[pypi-badge]: https://img.shields.io/pypi/v/weac.svg?logo=python&logoColor=white&color=f46b58&style=flat

[pypi-downloads-badge]: https://img.shields.io/pypi/dm/weac.svg?logo=python&logoColor=white&color=red&style=flat

[python-dist-badge]: https://img.shields.io/pypi/pyversions/weac.svg?style=flat&logo=python&logoColor=white

[doi-badge]: https://img.shields.io/badge/DOI-10.5281/zenodo.5773113-f03a6d.svg?style=flat
<!-- &logo=zenodo&logoColor=white -->

[release-badge]: https://img.shields.io/github/v/release/2phi/weac.svg?display_name=tag&color=f99a44&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAtCAYAAADsvzj/AAAACXBIWXMAAAsSAAALEgHS3X78AAAClUlEQVRoge1Z0W2DMBC9Vv0vGzQblE5QugEjZISMwAZNN2AEOkEzAtmAbkAmuMrSWbKezmBsC9IoT7KU4OPw8707G/PAzHQLeLwJFnciV4g7kWvDnci14WaIPCXeXxDRnohq51pHRC0RjYm+l8Gs7JGtYeaRdfTMXCT4tm0vviwGZm6ZeYe2sQ9oPQRcdAkESiCAGMUmiUjjcXxSrsdGIQR9KpEBHtKIjMoMRKZIjBJl1X+KrAYIL8ptzEiid/LCRZlCpJKGmka0C3PCVzhOTuEockZEa1p+uGTNAA7MXHvu9yV2D3YHp2/ITURL/hPYuESxdGr324FiCXfz85iTiCYpLI2ofbdvNyGpcoZwcvmdG8R+YhYH6POtR83MhGHEo4kUHl0fwA6j0cJEaBhBUoVS8rHYRBHxkdCqFNZ9N1q+3GhmnnXUxhVDBAenhloplQyJjrNsYaOhbVO8e7ilkdA07XOuLXC2r/aQsFGtlPxDyS0mspNBaTPoI6Hp2k10X5LXsFa4JLCKBRPBLXQIiVIGqVUzV35T2//FJEzTXqwKeTl6D3ip6pz/c/YWFRE9e/pe4f9F7Ps5p0iklMG9BAzoJdAOUQfancV2CLKGEGl7ppw4TMgKZbjoDTP08OGXiN6I6IGIPuR/DD4nZGXxJXJa9M6Pp/GDIpdvOWBAx7W00tH2WXz0kkOVonsfTD4Yf6eoKZqo/Z22FYhoWjlFdKmHFWt9H6mkiGiyOktUk7DWAZ2Ry9HT1+R4wJpfrExUfrQx5HC+9ZHpdy5HWxOJq1AK1iSyU651yrUobEnkN3j7EYAtpZUtGrQxkWz5QSsTwUXv30akcH5nK7sWW0jrIl+0siL109sSmJwwu2KzJcn7WY6I/gB+kRV89venQwAAAABJRU5ErkJggg==

[weac-badge]: https://img.shields.io/badge/weac-2.1-orange.svg?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAtCAYAAADsvzj/AAAACXBIWXMAAAsSAAALEgHS3X78AAAClUlEQVRoge1Z0W2DMBC9Vv0vGzQblE5QugEjZISMwAZNN2AEOkEzAtmAbkAmuMrSWbKezmBsC9IoT7KU4OPw8707G/PAzHQLeLwJFnciV4g7kWvDnci14WaIPCXeXxDRnohq51pHRC0RjYm+l8Gs7JGtYeaRdfTMXCT4tm0vviwGZm6ZeYe2sQ9oPQRcdAkESiCAGMUmiUjjcXxSrsdGIQR9KpEBHtKIjMoMRKZIjBJl1X+KrAYIL8ptzEiid/LCRZlCpJKGmka0C3PCVzhOTuEockZEa1p+uGTNAA7MXHvu9yV2D3YHp2/ITURL/hPYuESxdGr324FiCXfz85iTiCYpLI2ofbdvNyGpcoZwcvmdG8R+YhYH6POtR83MhGHEo4kUHl0fwA6j0cJEaBhBUoVS8rHYRBHxkdCqFNZ9N1q+3GhmnnXUxhVDBAenhloplQyJjrNsYaOhbVO8e7ilkdA07XOuLXC2r/aQsFGtlPxDyS0mspNBaTPoI6Hp2k10X5LXsFa4JLCKBRPBLXQIiVIGqVUzV35T2//FJEzTXqwKeTl6D3ip6pz/c/YWFRE9e/pe4f9F7Ps5p0iklMG9BAzoJdAOUQfancV2CLKGEGl7ppw4TMgKZbjoDTP08OGXiN6I6IGIPuR/DD4nZGXxJXJa9M6Pp/GDIpdvOWBAx7W00tH2WXz0kkOVonsfTD4Yf6eoKZqo/Z22FYhoWjlFdKmHFWt9H6mkiGiyOktUk7DWAZ2Ry9HT1+R4wJpfrExUfrQx5HC+9ZHpdy5HWxOJq1AK1iSyU651yrUobEnkN3j7EYAtpZUtGrQxkWz5QSsTwUXv30akcH5nK7sWW0jrIl+0siL109sSmJwwu2KzJcn7WY6I/gB+kRV89venQwAAAABJRU5ErkJggg==

[forks-url]: https://github.com/2phi/weac/network/members
[stars-url]: https://github.com/2phi/weac/stargazers
[contributors-url]: https://github.com/2phi/weac/graphs/contributors
[issues-url]: https://github.com/2phi/weac/issues
[pypi-url]: https://pypi.org/project/weac/
[release-url]: https://github.com/2phi/weac/releases
[weac-url]: https://github.com/2phi/weac/
[doi-url]: https://zenodo.org/badge/latestdoi/203163531