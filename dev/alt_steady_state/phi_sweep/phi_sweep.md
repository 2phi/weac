# Inclination (φ) sweep — tensile ease & ERR

Phis: [0.0, 15.0, 25.0, 35.0, 45.0]
Cases: ['case_1', 'case_5', 'case_12', 'case_21']

Expectation: higher φ → easier tensile failure + higher ERR.

## pst_fixed_cut

| Case/setup | max_Sxx_norm (increase) | thickness_fraction_without_density_gate (increase) | ERR (increase) |
|---|---|---|---|
| case_1/a | mixed | mixed | fail |
| case_1/b | mixed | pass | mixed |
| case_5/a | mixed | pass | mixed |
| case_5/b | mixed | pass | mixed |
| case_12/a | mixed | fail | fail |
| case_12/b | mixed | pass | mixed |
| case_21/a | mixed | mixed | mixed |
| case_21/b | mixed | fail | mixed |

## pst_critical_cut

| Case/setup | characteristic_length (decrease) | ERR (increase) |
|---|---|---|
| case_1/a | mixed | pass |
| case_1/b | mixed | pass |
| case_5/a | pass | pass |
| case_5/b | mixed | pass |
| case_12/a | mixed | mixed |
| case_12/b | pass | pass |
| case_21/a | pass | fail |
| case_21/b | mixed | pass |

## pst_critical_mass

| Case/setup | critical_mass_kg (decrease) | ERR (increase) |
|---|---|---|
| case_1/a | pass | pass |
| case_1/b | pass | mixed |
| case_5/a | pass | mixed |
| case_5/b | pass | fail |
| case_12/a | pass | pass |
| case_12/b | pass | mixed |
| case_21/a | pass | fail |
| case_21/b | pass | fail |

## pst_touchdown_cut

| Case/setup | thickness_fraction_without_density_gate (increase) | ERR (increase) |
|---|---|---|
| case_1/a | pass | fail |
| case_1/b | pass | mixed |
| case_5/a | pass | mixed |
| case_5/b | pass | mixed |
| case_12/a | pass | fail |
| case_12/b | pass | mixed |
| case_21/a | pass | mixed |
| case_21/b | flat | mixed |

## Tallies

| Method | Ease metric | pass | fail | mixed | flat | insuff |
|---|---|---|---|---|---|---|
| pst_fixed_cut | max_Sxx_norm (increase) | 0 | 0 | 8 | 0 | 0 |
| pst_fixed_cut | thickness_fraction_without_density_gate (increase) | 4 | 2 | 2 | 0 | 0 |
| pst_fixed_cut | ERR (increase) | 0 | 2 | 6 | 0 | 0 |
| pst_critical_cut | characteristic_length (decrease) | 3 | 0 | 5 | 0 | 0 |
| pst_critical_cut | ERR (increase) | 6 | 1 | 1 | 0 | 0 |
| pst_critical_mass | critical_mass_kg (decrease) | 8 | 0 | 0 | 0 | 0 |
| pst_critical_mass | ERR (increase) | 2 | 3 | 3 | 0 | 0 |
| pst_touchdown_cut | thickness_fraction_without_density_gate (increase) | 7 | 0 | 0 | 1 | 0 |
| pst_touchdown_cut | ERR (increase) | 0 | 2 | 6 | 0 | 0 |

## Numeric series

### pst_fixed_cut

**case_1/a**

| φ | max_Sxx_norm | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|---|
| 0 | 13.73 | 0.8159 | 4.767 | upslope |
| 15 | 13.73 | 0.8308 | 4.702 | upslope |
| 25 | 13.21 | 0.8358 | 4.32 | upslope |
| 35 | 12.29 | 0.8308 | 3.715 | upslope |
| 45 | 11 | 0.8259 | 2.961 | upslope |

**case_1/b**

| φ | max_Sxx_norm | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|---|
| 0 | 6.864 | 0.6708 | 3.529 | upslope |
| 15 | 7.103 | 0.7207 | 3.829 | upslope |
| 25 | 6.993 | 0.7406 | 3.767 | upslope |
| 35 | 6.671 | 0.7506 | 3.502 | upslope |
| 45 | 6.146 | 0.7556 | 3.065 | upslope |

**case_5/a**

| φ | max_Sxx_norm | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|---|
| 0 | 5.491 | 0.6028 | 3.412 | upslope |
| 15 | 5.777 | 0.6766 | 3.902 | upslope |
| 25 | 5.749 | 0.7046 | 3.976 | upslope |
| 35 | 5.546 | 0.7226 | 3.835 | upslope |
| 45 | 5.175 | 0.7345 | 3.495 | upslope |

**case_5/b**

| φ | max_Sxx_norm | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|---|
| 0 | 2.7 | 0.3194 | 1.027 | downslope |
| 15 | 2.84 | 0.3792 | 1.169 | upslope |
| 25 | 2.826 | 0.4112 | 1.19 | upslope |
| 35 | 2.727 | 0.4351 | 1.149 | upslope |
| 45 | 2.544 | 0.4471 | 1.049 | upslope |

**case_12/a**

| φ | max_Sxx_norm | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|---|
| 0 | 4.355 | 0.1673 | 0.8298 | upslope |
| 15 | 4.377 | 0.1594 | 0.7874 | downslope |
| 25 | 4.225 | 0.1534 | 0.7191 | downslope |
| 35 | 3.945 | 0.1434 | 0.6279 | downslope |
| 45 | 3.545 | 0.1335 | 0.5248 | downslope |

**case_12/b**

| φ | max_Sxx_norm | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|---|
| 0 | 5.491 | 0.6028 | 3.412 | upslope |
| 15 | 5.777 | 0.6766 | 3.902 | upslope |
| 25 | 5.749 | 0.7046 | 3.976 | upslope |
| 35 | 5.546 | 0.7226 | 3.835 | upslope |
| 45 | 5.175 | 0.7345 | 3.495 | upslope |

**case_21/a**

| φ | max_Sxx_norm | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|---|
| 0 | 2.08 | 0.4712 | 0.4569 | downslope |
| 15 | 2.35 | 0.5265 | 0.5435 | upslope |
| 25 | 2.441 | 0.5442 | 0.5643 | upslope |
| 35 | 2.459 | 0.5487 | 0.5523 | upslope |
| 45 | 2.402 | 0.5442 | 0.5087 | upslope |

**case_21/b**

| φ | max_Sxx_norm | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|---|
| 0 | 1.654 | 0.03761 | 0.2727 | upslope |
| 15 | 1.654 | 0.0354 | 0.2868 | downslope |
| 25 | 1.592 | 0.03097 | 0.2806 | downslope |
| 35 | 1.481 | 0.02434 | 0.2626 | downslope |
| 45 | 1.325 | 0.0177 | 0.235 | downslope |

### pst_critical_cut

**case_1/a**

| φ | characteristic_length | ERR | winner |
|---|---|---|---|
| 0 | 134.8 | 0.08003 | upslope |
| 15 | 128.7 | 0.09193 | upslope |
| 25 | 127.2 | 0.09872 | upslope |
| 35 | 127.7 | 0.1036 | upslope |
| 45 | 130.8 | 0.1062 | upslope |

**case_1/b**

| φ | characteristic_length | ERR | winner |
|---|---|---|---|
| 0 | 190.9 | 0.2368 | upslope |
| 15 | 177.2 | 0.3024 | upslope |
| 25 | 171.8 | 0.3444 | upslope |
| 35 | 169.5 | 0.3813 | upslope |
| 45 | 170.1 | 0.4081 | upslope |

**case_5/a**

| φ | characteristic_length | ERR | winner |
|---|---|---|---|
| 0 | 213.4 | 0.342 | upslope |
| 15 | 196 | 0.456 | upslope |
| 25 | 188.7 | 0.5316 | upslope |
| 35 | 184.8 | 0.5996 | upslope |
| 45 | 184.1 | 0.6517 | upslope |

**case_5/b**

| φ | characteristic_length | ERR | winner |
|---|---|---|---|
| 0 | 304.3 | 0.2717 | downslope |
| 15 | 288.1 | 0.3208 | upslope |
| 25 | 283.3 | 0.3507 | upslope |
| 35 | 282.8 | 0.3736 | upslope |
| 45 | 284.8 | 0.3818 | upslope |

**case_12/a**

| φ | characteristic_length | ERR | winner |
|---|---|---|---|
| 0 | 239.6 | 0.09376 | upslope |
| 15 | 233.8 | 0.06538 | downslope |
| 25 | 234.7 | 0.0559 | downslope |
| 35 | 239.6 | 0.05449 | downslope |
| 45 | 249.7 | 0.0613 | downslope |

**case_12/b**

| φ | characteristic_length | ERR | winner |
|---|---|---|---|
| 0 | 213.4 | 0.342 | upslope |
| 15 | 196 | 0.456 | upslope |
| 25 | 188.7 | 0.5316 | upslope |
| 35 | 184.8 | 0.5996 | upslope |
| 45 | 184.1 | 0.6517 | upslope |

**case_21/a**

| φ | characteristic_length | ERR | winner |
|---|---|---|---|
| 0 | 338.5 | 0.1533 | downslope |
| 15 | 302.6 | 0.1518 | upslope |
| 25 | 286.5 | 0.1494 | upslope |
| 35 | 275.8 | 0.1456 | upslope |
| 45 | 270.8 | 0.1401 | upslope |

**case_21/b**

| φ | characteristic_length | ERR | winner |
|---|---|---|---|
| 0 | 388.8 | 0.1277 | downslope |
| 15 | 386.8 | 0.1293 | downslope |
| 25 | 393.3 | 0.1332 | downslope |
| 35 | 406.9 | 0.1399 | downslope |
| 45 | 430.4 | 0.1507 | downslope |

### pst_critical_mass

**case_1/a**

| φ | critical_mass_kg | ERR | winner |
|---|---|---|---|
| 0 | 0.6165 | 0.08469 | upslope |
| 15 | 0.3899 | 0.0898 | upslope |
| 25 | 0.322 | 0.09419 | upslope |
| 35 | 0.2922 | 0.09742 | upslope |
| 45 | 0.2899 | 0.09825 | upslope |

**case_1/b**

| φ | critical_mass_kg | ERR | winner |
|---|---|---|---|
| 0 | 3.964 | 0.3127 | upslope |
| 15 | 2.109 | 0.2886 | upslope |
| 25 | 1.599 | 0.3071 | upslope |
| 35 | 1.31 | 0.3296 | upslope |
| 45 | 1.147 | 0.3478 | upslope |

**case_5/a**

| φ | critical_mass_kg | ERR | winner |
|---|---|---|---|
| 0 | 6.665 | 0.5062 | upslope |
| 15 | 3.232 | 0.4295 | upslope |
| 25 | 2.377 | 0.461 | upslope |
| 35 | 1.901 | 0.504 | upslope |
| 45 | 1.625 | 0.5425 | upslope |

**case_5/b**

| φ | critical_mass_kg | ERR | winner |
|---|---|---|---|
| 0 | 25.82 | 0.5629 | upslope |
| 15 | 13.43 | 0.3555 | upslope |
| 25 | 10.32 | 0.3274 | upslope |
| 35 | 8.557 | 0.3174 | upslope |
| 45 | 7.513 | 0.3112 | upslope |

**case_12/a**

| φ | critical_mass_kg | ERR | winner |
|---|---|---|---|
| 0 | 13.64 | 0.1404 | upslope |
| 15 | 10.35 | 0.1501 | upslope |
| 25 | 9.36 | 0.1627 | upslope |
| 35 | 8.859 | 0.175 | upslope |
| 45 | 8.704 | 0.184 | upslope |

**case_12/b**

| φ | critical_mass_kg | ERR | winner |
|---|---|---|---|
| 0 | 6.665 | 0.5062 | upslope |
| 15 | 3.232 | 0.4295 | upslope |
| 25 | 2.377 | 0.461 | upslope |
| 35 | 1.901 | 0.504 | upslope |
| 45 | 1.625 | 0.5425 | upslope |

**case_21/a**

| φ | critical_mass_kg | ERR | winner |
|---|---|---|---|
| 0 | 30.13 | 0.2387 | upslope |
| 15 | 15.44 | 0.1612 | upslope |
| 25 | 11.52 | 0.1439 | upslope |
| 35 | 9.277 | 0.1338 | upslope |
| 45 | 7.916 | 0.1259 | upslope |

**case_21/b**

| φ | critical_mass_kg | ERR | winner |
|---|---|---|---|
| 0 | 47.66 | 0.2714 | upslope |
| 15 | 35.16 | 0.1933 | upslope |
| 25 | 31.05 | 0.1668 | upslope |
| 35 | 28.69 | 0.1476 | upslope |
| 45 | 27.49 | 0.1314 | upslope |

### pst_touchdown_cut

**case_1/a**

| φ | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|
| 0 | 0.4627 | 0.6443 | downslope |
| 15 | 0.5174 | 0.6396 | upslope |
| 25 | 0.5473 | 0.6169 | upslope |
| 35 | 0.5672 | 0.5783 | upslope |
| 45 | 0.5871 | 0.5253 | upslope |

**case_1/b**

| φ | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|
| 0 | 0.3791 | 1.179 | downslope |
| 15 | 0.4464 | 1.188 | upslope |
| 25 | 0.4838 | 1.173 | upslope |
| 35 | 0.5112 | 1.134 | upslope |
| 45 | 0.5387 | 1.071 | upslope |

**case_5/a**

| φ | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|
| 0 | 0.3413 | 1.436 | downslope |
| 15 | 0.4092 | 1.463 | upslope |
| 25 | 0.4451 | 1.461 | upslope |
| 35 | 0.479 | 1.436 | upslope |
| 45 | 0.507 | 1.383 | upslope |

**case_5/b**

| φ | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|
| 0 | 0.475 | 2.18 | downslope |
| 15 | 0.523 | 2.184 | upslope |
| 25 | 0.5469 | 2.125 | upslope |
| 35 | 0.5669 | 2.016 | upslope |
| 45 | 0.5808 | 1.856 | upslope |

**case_12/a**

| φ | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|
| 0 | 0.1873 | 1.957 | upslope |
| 15 | 0.2809 | 1.875 | upslope |
| 25 | 0.3446 | 1.771 | upslope |
| 35 | 0.3964 | 1.632 | upslope |
| 45 | 0.4004 | 1.461 | upslope |

**case_12/b**

| φ | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|
| 0 | 0.3413 | 1.436 | downslope |
| 15 | 0.4092 | 1.463 | upslope |
| 25 | 0.4451 | 1.461 | upslope |
| 35 | 0.479 | 1.436 | upslope |
| 45 | 0.507 | 1.383 | upslope |

**case_21/a**

| φ | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|
| 0 | 0.7522 | 2.259 | downslope |
| 15 | 0.7655 | 2.303 | upslope |
| 25 | 0.7743 | 2.252 | upslope |
| 35 | 0.7765 | 2.133 | upslope |
| 45 | 0.7788 | 1.949 | upslope |

**case_21/b**

| φ | thickness_fraction_without_density_gate | ERR | winner |
|---|---|---|---|
| 0 | 0.1128 | 2.264 | downslope |
| 15 | 0.1128 | 2.284 | downslope |
| 25 | 0.1128 | 1.949 | upslope |
| 35 | 0.1128 | 1.74 | upslope |
| 45 | 0.1128 | 1.494 | upslope |

