# BPM Model

This is the source code for the bipolar membrane (BPM) model published in the article:

> Cassady, H. J.; Rochow, M. F.; Hickner, M. A. The Impact of Membrane Orientation on Ion Flux in Bipolar Membranes. *Journal of Membrane Science* **2024**, *702*, 122748. DOI: [10.1016/j.memsci.2024.122748](https://doi.org/10.1016/j.memsci.2024.122748).

An example Jupyter notebook that generates the figures in the article is included in this repository.

## Getting Started

Import the model and get the pint unit registry from the imported model:

```python
import bpmModel
u = bpmModel.u
```

### Setting up the model

The BPM model is comprised of two membrane objects: an anion exchange membrane (AEM) and a cation exchange membrane (CEM). Creating the two membrane objects:

```python
N212 = bpmModel.IonExchangeMembrane(
    name="Nafion 212",
    membrane_type="CEM",
    thickness=50 * u["µm"],
    fixed_charge_concentration=1.6 * u["M"],
    cation_diffusion_coefficient=1.0e-6 * u["cm²/s"],
    anion_diffusion_coefficient=5.0e-8 * u["cm²/s"],
)
A40 = bpmModel.IonExchangeMembrane(
    name="A40",
    membrane_type="AEM",
    thickness=40 * u["µm"],
    fixed_charge_concentration=2.65 * u["M"],
    cation_diffusion_coefficient=3.0e-7 * u["cm²/s"],
    anion_diffusion_coefficient=1.0e-6 * u["cm²/s"],
)
```

The BPM model is initialized by providing these two membranes, as well as a left concentration and a right concentration:

```python
model = bpmModel.BPMSystem(
    CEM=N212, AEM=A40, left_solution_concentration=0.5, right_solution_concentration=0.0
)
```

### Using the model

The model can be used to either generate the concentration profile through the system or to investigate the impact of changing the membrane parameters on the flux differential. 

#### Concentration Profile

To generate a concentration profile, a sub-model can be extracted by specifying one of the membrane positions (`AEM_high` / `CEM_high` / `AEM_low` / `CEM_low`):

```python
model.AEM_high
```

The concentration profile can be generated with:

```python
AEM_high = model.AEM_high.generate_concentration_profiles(show_Cs=True)
```

This function returns a dictionary with the figure and data frames containing the results of the model. The theoretical interface concentration, $C_s$ for a given orientation is included as a parameter of the sub-model:

```python
model.AEM_low.interface_concentration
```

#### Flux Differential

The flux differential can be computed as a function of either the `fixed_charge_concentration` or the `ion_diffusion_coefficients_counter_co`:

```python
fig, ax, data = model.generate_differential_plot(
    "fixed_charge_concentration",
    x_range=[1, 4],
    y_range=[1, 4],
    mesh_type="linear",
    n_points=20,
    text_location="off",
    contourf_args={"cmap": cmap},
    colorbar_args={
        "label": r"$\lambda_J$: Flux Differential",
    },
    auto_vminmax=True,
)
```

The plot for the flux differential is created from a $n \times n$ grid, with the flux differential computed at each point. The grid size is set by the `n_points` parameter. Increasing the value of `n_points` increases the fidelity of the plot, but the runtime increases with $n^2$. Make sure to adjust this parameter to balance the resolution/time trade-off when running the model.