<!-- LOGO AND TITLE-->
<p align="right"><img src="https://github.com/2phi/weac/raw/main/img/logo.png" alt="Logo" width="80" height="80"></p>

# WEAC &nbsp;·&nbsp; Weak Layer Anticrack Nucleation Model

<!-- PROJECT SHIELDS -->
<!-- PyPI Downloads /pypi/:period/:packageName -->
<!-- PyPI Versions /pypi/pyversions/:packageName -->
<!-- [![Forks][forks-shield]][forks-url] -->
[![Weac][weac-shield]][weac-url]
[![Release][release-shield]][release-url]
[![PyPI][pypi-shield]][pypi-url]
[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![DOI](https://zenodo.org/badge/203163531.svg)](https://zenodo.org/badge/latestdoi/203163531)\
Implementation of closed-form analytical models for the analysis of dry-snow slab avalanche release.

[View demo](https://github.com/2phi/weac/blob/main/demo/demo.ipynb) · 
[Report bug](https://github.com/2phi/weac/issues) · 
[Request feature](https://github.com/2phi/weac/issues) · 
[Read the docs](https://2phi.github.io/weac/)

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

Written in [Python](https://www.python.org) and built with [Atom](https://atom.io), [GitKraken](https://www.gitkraken.com), and [Jupyter](https://jupyter.org). Note that [release v1.0](https://github.com/2phi/weac/releases/tag/v1.0.0) was written and built in [MATLAB](https://www.mathworks.com/products/matlab.html).

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
Choose a profile from the database (see [demo](https://github.com/2phi/weac/blob/main/demo/demo.ipynb)) or create your own.
```python
myprofile = [[180, 100],
             [190,  40],
             [230, 130],
             [380,  20],
             [270, 100]]
```
Create a model instance with optional custom layering.
```python
skier = weac.Layered(system='skier', layers=myprofile)
```
Calculate lists of segment lengths, locations of foundations, and position and magnitude of skier loads from the inputs total length `L` (mm), crack length `a` (mm), and skier weight `m` (kg). We can choose to analyze the situtation before a crack appears even if a crack length > 0 is set by replacing the `'crack'` key thorugh the `'uncracked'` key.
```python
segments = skier.calc_segments(L=10000, a=300, m=80)['crack']
```
Assemble the system of linear equations and solve the boundary-value problem for the free constants `C` providing the inclination `phi` in degrees.
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

# Plot slab displacements
weac.plot.displacements(skier, x=xq, z=zq, **segments)

# Plot weak-layer stresses
weac.plot.stresses(skier, x=xb, z=zq, **segments)
```

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/2phi/weac/issues) for a list of proposed features and known issues.

### v2.2
- [ ] Finite fracture mechanics implementation for layered snow covers

## Release history

### v2.1
- Consistent use of coordinate system with downward pointing z-axis
- Consitent top-to-bottom ordering of slab layers
- Implementation of PSTs cut from either left or right side

### v2.0
- Completely rewritten in Python
- Coupled bending-extension ODE solver implemented
- Stress analysis of arbitrarily layered snow slabs
- FEM validation of
  - displacements
  - weak-layer stresses
  - energy release rates in weak layers
- Documentation
- Demo and examples

### v1.0

- Written in MATLAB
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
[contributors-shield]: https://img.shields.io/github/contributors/2phi/weac.svg?style=flat&logo=github
[contributors-url]: https://github.com/2phi/weac/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/2phi/weac.svg?&color=blueviolet&style=flat&logo=github
[forks-url]: https://github.com/2phi/weac/network/members
[stars-shield]: https://img.shields.io/github/stars/2phi/weac.svg?style=flat&logo=github&color=red
[stars-url]: https://github.com/2phi/weac/stargazers
[issues-shield]: https://img.shields.io/github/issues/2phi/weac.svg?style=flat&logo=github
[issues-url]: https://github.com/2phi/weac/issues
[pypi-shield]: https://img.shields.io/pypi/v/weac.svg?logo=pypi&logoColor=white&color=blue
[pypi-url]: https://pypi.org/project/weac/
[release-shield]: https://img.shields.io/github/v/release/2phi/weac.svg?display_name=tag&color=blueviolet&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAtCAYAAADsvzj/AAAACXBIWXMAAAsSAAALEgHS3X78AAAClUlEQVRoge1Z0W2DMBC9Vv0vGzQblE5QugEjZISMwAZNN2AEOkEzAtmAbkAmuMrSWbKezmBsC9IoT7KU4OPw8707G/PAzHQLeLwJFnciV4g7kWvDnci14WaIPCXeXxDRnohq51pHRC0RjYm+l8Gs7JGtYeaRdfTMXCT4tm0vviwGZm6ZeYe2sQ9oPQRcdAkESiCAGMUmiUjjcXxSrsdGIQR9KpEBHtKIjMoMRKZIjBJl1X+KrAYIL8ptzEiid/LCRZlCpJKGmka0C3PCVzhOTuEockZEa1p+uGTNAA7MXHvu9yV2D3YHp2/ITURL/hPYuESxdGr324FiCXfz85iTiCYpLI2ofbdvNyGpcoZwcvmdG8R+YhYH6POtR83MhGHEo4kUHl0fwA6j0cJEaBhBUoVS8rHYRBHxkdCqFNZ9N1q+3GhmnnXUxhVDBAenhloplQyJjrNsYaOhbVO8e7ilkdA07XOuLXC2r/aQsFGtlPxDyS0mspNBaTPoI6Hp2k10X5LXsFa4JLCKBRPBLXQIiVIGqVUzV35T2//FJEzTXqwKeTl6D3ip6pz/c/YWFRE9e/pe4f9F7Ps5p0iklMG9BAzoJdAOUQfancV2CLKGEGl7ppw4TMgKZbjoDTP08OGXiN6I6IGIPuR/DD4nZGXxJXJa9M6Pp/GDIpdvOWBAx7W00tH2WXz0kkOVonsfTD4Yf6eoKZqo/Z22FYhoWjlFdKmHFWt9H6mkiGiyOktUk7DWAZ2Ry9HT1+R4wJpfrExUfrQx5HC+9ZHpdy5HWxOJq1AK1iSyU651yrUobEnkN3j7EYAtpZUtGrQxkWz5QSsTwUXv30akcH5nK7sWW0jrIl+0siL109sSmJwwu2KzJcn7WY6I/gB+kRV89venQwAAAABJRU5ErkJggg==
[release-url]: https://github.com/2phi/weac/releases
[weac-shield]: https://img.shields.io/badge/weac-2.0-orange.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAtCAYAAADsvzj/AAAACXBIWXMAAAsSAAALEgHS3X78AAAClUlEQVRoge1Z0W2DMBC9Vv0vGzQblE5QugEjZISMwAZNN2AEOkEzAtmAbkAmuMrSWbKezmBsC9IoT7KU4OPw8707G/PAzHQLeLwJFnciV4g7kWvDnci14WaIPCXeXxDRnohq51pHRC0RjYm+l8Gs7JGtYeaRdfTMXCT4tm0vviwGZm6ZeYe2sQ9oPQRcdAkESiCAGMUmiUjjcXxSrsdGIQR9KpEBHtKIjMoMRKZIjBJl1X+KrAYIL8ptzEiid/LCRZlCpJKGmka0C3PCVzhOTuEockZEa1p+uGTNAA7MXHvu9yV2D3YHp2/ITURL/hPYuESxdGr324FiCXfz85iTiCYpLI2ofbdvNyGpcoZwcvmdG8R+YhYH6POtR83MhGHEo4kUHl0fwA6j0cJEaBhBUoVS8rHYRBHxkdCqFNZ9N1q+3GhmnnXUxhVDBAenhloplQyJjrNsYaOhbVO8e7ilkdA07XOuLXC2r/aQsFGtlPxDyS0mspNBaTPoI6Hp2k10X5LXsFa4JLCKBRPBLXQIiVIGqVUzV35T2//FJEzTXqwKeTl6D3ip6pz/c/YWFRE9e/pe4f9F7Ps5p0iklMG9BAzoJdAOUQfancV2CLKGEGl7ppw4TMgKZbjoDTP08OGXiN6I6IGIPuR/DD4nZGXxJXJa9M6Pp/GDIpdvOWBAx7W00tH2WXz0kkOVonsfTD4Yf6eoKZqo/Z22FYhoWjlFdKmHFWt9H6mkiGiyOktUk7DWAZ2Ry9HT1+R4wJpfrExUfrQx5HC+9ZHpdy5HWxOJq1AK1iSyU651yrUobEnkN3j7EYAtpZUtGrQxkWz5QSsTwUXv30akcH5nK7sWW0jrIl+0siL109sSmJwwu2KzJcn7WY6I/gB+kRV89venQwAAAABJRU5ErkJggg==
[weac-url]: https://github.com/2phi/weac/
