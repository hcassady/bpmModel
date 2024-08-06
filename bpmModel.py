import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import pint_pandas
from scipy.constants import physical_constants
from scipy.optimize import fsolve, root

# Setup pint:
u = pint.UnitRegistry()
pint_pandas.PintType.ureg = u
pint_pandas.PintType.ureg.setup_matplotlib()
u.Unit.default_format = "~P"
pint_pandas.PintType.ureg.default_format = "~P"
warnings.filterwarnings("ignore", category=pint.UnitStrippedWarning)


# Define physical constants:
def get_pint_constant(constant_name):
    """Extract a given scipy physical constant and return with pint units."""
    value, unit, *_ = physical_constants[constant_name]
    return value * u[unit]


F = get_pint_constant("Faraday constant")
kb = get_pint_constant("Boltzmann constant")
R = get_pint_constant("molar gas constant")
q = get_pint_constant("elementary charge")
T = u.Quantity(20, u.degC).to(u.K)


class IonExchangeMembrane:
    def __init__(
        self,
        name,
        membrane_type,
        thickness,
        fixed_charge_concentration,
        cation_diffusion_coefficient,
        anion_diffusion_coefficient,
    ):
        self.parameters = {}
        self.update_parameters(
            name,
            membrane_type,
            thickness,
            fixed_charge_concentration,
            cation_diffusion_coefficient,
            anion_diffusion_coefficient,
        )

    def __repr__(self):
        return (
            f"Type: {self.type}\n"
            f"Name: {self.name}\n"
            f"Thickness: {self.thickness}\n"
            f"Fixed Charge Concentration:{self.fixed_charge_concentration}\n"
            f"Cation Diffusion Coefficient: {self.cation_diffusion_coefficient}\n"
            f"Anion Diffusion Coefficient {self.anion_diffusion_coefficient}\n"
        )

    def flux_equation(self, x, C0, C1):
        return self.flux_equation_base(
            C0=C0,
            C1=C1,
            D_cat=self.cation_diffusion_coefficient,
            D_an=self.anion_diffusion_coefficient,
            zm=self.charge,
            Xm=self.fixed_charge_concentration,
            x=x,
        )

    def flux_equation_base(self, C0, C1, D_cat, D_an, zm, Xm, x):
        r = D_cat / D_an
        t0 = (-1 * D_cat) / (x * (r + 1) ** 2)
        t1 = (zm * Xm) * (r - 1)
        t2 = np.log(((zm * Xm) + (r + 1) * C1) / ((zm * Xm) + (r + 1) * C0))
        t3 = 2 * (r + 1) * (C1 - C0)
        flux = t0 * (t1 * t2 + t3)
        return flux

    def update_parameters(
        self,
        name=None,
        membrane_type=None,
        thickness=None,
        fixed_charge_concentration=None,
        cation_diffusion_coefficient=None,
        anion_diffusion_coefficient=None,
    ):
        if name:
            self.name = name
            self.parameters["name"] = name
        if membrane_type:
            self.type = membrane_type
            self.parameters["membrane_type"] = membrane_type
            self.charge = {"AEM": 1, "CEM": -1}[self.type]
        if thickness:
            self.thickness = thickness
            self.parameters["thickness"] = thickness
            self.t = self.thickness
        if fixed_charge_concentration:
            self.fixed_charge_concentration = fixed_charge_concentration
            self.parameters["fixed_charge_concentration"] = fixed_charge_concentration
        if cation_diffusion_coefficient:
            self.cation_diffusion_coefficient = cation_diffusion_coefficient
            self.parameters["cation_diffusion_coefficient"] = (
                cation_diffusion_coefficient
            )
            self.cation_mobility = cation_diffusion_coefficient / (kb * T)
        if anion_diffusion_coefficient:
            self.anion_diffusion_coefficient = anion_diffusion_coefficient
            self.parameters["anion_diffusion_coefficient"] = anion_diffusion_coefficient
            self.anion_mobility = anion_diffusion_coefficient / (kb * T)

    def compute_donnan_layer_concentration(self, solution_concentration):
        C = solution_concentration
        Xm = self.fixed_charge_concentration
        ω = self.charge
        return 0.5 * (np.sqrt(Xm**2 + 4 * C**2) - ω * Xm)

    def compute_anion_concentration(self, cation_concentration):
        Xm = self.fixed_charge_concentration
        ω = self.charge

        return cation_concentration + (ω * Xm)

    def compute_salt_concentration_from_cation_concentration(
        self, cation_concentration
    ):
        if self.type == "AEM":
            return cation_concentration
        else:
            return self.compute_anion_concentration(cation_concentration)

    def compute_membrane_flux(
        self, left_solution_concentration, right_solution_concentration
    ):
        """This function computes the flux given the concentration at the boundary
        outside of the membrane. This includes the concentration change across the
        Donnan layer."""
        left_cation_concentration = self.compute_donnan_layer_concentration(
            left_solution_concentration
        )
        right_cation_concentration = self.compute_donnan_layer_concentration(
            right_solution_concentration
        )
        return self.flux_equation(
            self.thickness, left_cation_concentration, right_cation_concentration
        )

    def compute_cation_concentration_at_point(
        self, position, flux, left_solution_concentration, right_solution_concentration
    ):
        """Return the cation concentration at a given position and flux. The
        left and right concentrations outside of the membrane are also required,
        in order to set the boundary conditions and properly set the guess for
        the solver to work properly."""
        left_cation_concentration = self.compute_donnan_layer_concentration(
            left_solution_concentration
        )
        right_cation_concentration = self.compute_donnan_layer_concentration(
            right_solution_concentration
        )
        J0 = flux.to_base_units().magnitude * 1e8

        def fx(C):
            J1 = (
                self.flux_equation(position, left_cation_concentration, C * u["M"])
                .to_base_units()
                .magnitude
                * 1e8
            )
            return J1 - J0

        vfx = np.vectorize(fx)
        guess = (left_cation_concentration + right_cation_concentration) / 2
        result = fsolve(vfx, guess)
        return result

    def compute_membrane_concentration_profile(
        self,
        flux,
        left_solution_concentration,
        right_solution_concentration,
        n_points=200,
    ):
        """Gives a dataframe that has the membrane concentration profile."""

        # Start position is small value, so solver is not solving exactly at the
        # boundary. The end position is also reduced by this value. This avoids
        # numerical issues related to solutions at the boundary.
        start_position = 1e-3
        end_position = self.t.to("µm").magnitude - start_position
        x_values = np.linspace(start_position, end_position, n_points)

        cation_concentrations = np.array([])
        for x in x_values:
            cation_concentrations = np.append(
                cation_concentrations,
                self.compute_cation_concentration_at_point(
                    x * u["µm"],
                    flux,
                    left_solution_concentration.to("M"),
                    right_solution_concentration.to("M"),
                ),
            )

        C_cat = cation_concentrations * u["M"]
        position = x_values * u["µm"]
        C_an = C_cat + self.charge * self.fixed_charge_concentration

        if self.type == "AEM":
            C_co = C_cat
            C_counter = C_an
        elif self.type == "CEM":
            C_co = C_an
            C_counter = C_cat

        df = pd.DataFrame(
            data={
                "x": pd.Series(position, dtype=f"pint[{position.units}]"),
                "C_cat": pd.Series(C_cat, dtype=f"pint[{C_cat.units}]"),
                "C_an": pd.Series(C_an, dtype=f"pint[{C_an.units}]"),
                "C_co": pd.Series(C_co, dtype=f"pint[{C_co.units}]"),
                "C_counter": pd.Series(C_counter, dtype=f"pint[{C_an.units}]"),
            }
        )

        return df


class BPMModel:
    def __init__(
        self,
        left_solution_concentration,
        right_solution_concentration,
        left_membrane_layer,
        right_membrane_layer,
    ):
        self.left_solution_concentration = left_solution_concentration
        self.right_solution_concentration = right_solution_concentration
        self.left_membrane_layer = left_membrane_layer
        self.right_membrane_layer = right_membrane_layer
        self.compute_bpm_flux()

    def update_membrane_layers(
        self, left_membrane_layer=None, right_membrane_layer=None
    ):
        if left_membrane_layer:
            self.left_membrane_layer = left_membrane_layer
        if right_membrane_layer:
            self.right_membrane_layer = right_membrane_layer
        self.compute_bpm_flux()

    def update_fixed_charge_concentrations(
        self, new_left_fixed_charge, new_right_fixed_charge
    ):
        self.left_membrane_layer.update_parameters(
            fixed_charge_concentration=new_left_fixed_charge
        )
        self.right_membrane_layer.update_parameters(
            fixed_charge_concentration=new_right_fixed_charge
        )
        self.compute_bpm_flux()

    def _calculate_bpm_flux(self):
        def fx(Cs):
            C0 = self.left_solution_concentration
            C1 = self.right_solution_concentration
            Cs = Cs * u["M"]
            left_flux = self.left_membrane_layer.compute_membrane_flux(C0, Cs)
            left_flux = left_flux.to("µmol/m²/s").magnitude
            right_flux = (
                self.right_membrane_layer.compute_membrane_flux(Cs, C1)
                .to("umol/m²/s")
                .magnitude
            )
            return left_flux - right_flux

        Cs_initial_guess = np.mean(
            [
                self.left_solution_concentration.magnitude,
                self.right_solution_concentration.magnitude,
            ]
        )
        Cs = root(fx, [Cs_initial_guess])
        Cs = Cs["x"] * u["M"]
        # Cs = Cs[0] * u["M"]
        flux = self.left_membrane_layer.compute_membrane_flux(
            self.left_solution_concentration, Cs
        ).to("mol/cm²/s")

        self.flux = flux
        self.interface_concentration = Cs

    def compute_bpm_flux(self):
        self._calculate_bpm_flux()
        return (self.flux, self.interface_concentration)

    def generate_concentration_profiles(self, n_points=30, show_Cs=False):
        left_concentration_profile = (
            self.left_membrane_layer.compute_membrane_concentration_profile(
                self.flux,
                self.left_solution_concentration,
                self.interface_concentration,
            )
        )
        left_concentration_profile.x = (
            left_concentration_profile.x - self.left_membrane_layer.t
        )
        right_concentration_profile = (
            self.right_membrane_layer.compute_membrane_concentration_profile(
                self.flux,
                self.interface_concentration,
                self.right_solution_concentration,
            )
        )

        membrane_limit = np.max(
            [
                self.left_membrane_layer.t.magnitude,
                self.right_membrane_layer.t.magnitude,
            ]
        )
        solution_region_width = membrane_limit * 1.5
        left_solution_domain = [
            -1 * solution_region_width,
            -1 * self.left_membrane_layer.t.magnitude,
        ]
        right_solution_domain = [
            solution_region_width,
            self.right_membrane_layer.t.magnitude,
        ]
        left_solution_concentrations = [
            self.left_solution_concentration.magnitude,
            self.left_solution_concentration.magnitude,
        ]
        right_solution_concentrations = [
            self.right_solution_concentration.magnitude,
            self.right_solution_concentration.magnitude,
        ]

        fig, ax = plt.subplots(figsize=(3.534, 2.953))

        ax.plot(
            left_solution_domain, left_solution_concentrations, "k-", linewidth=2
        )  # Plot LC concentration
        ax.plot(
            right_solution_domain, right_solution_concentrations, "k-", linewidth=2
        )  # Plot LC concentration
        ax.plot(
            left_concentration_profile.x,
            left_concentration_profile.C_co,
            "k-",
            linewidth=2,
        )
        ax.plot(
            right_concentration_profile.x,
            right_concentration_profile.C_co,
            "k-",
            linewidth=2,
        )

        y_lims = {
            "ymin": -0.05,
            "ymax": 1.1
            * np.max([left_solution_concentrations, right_solution_concentrations]),
        }
        x_lines = [left_solution_domain[1], 0, right_solution_domain[1]]
        ax.vlines(x_lines, **y_lims, colors="black", linestyle="dashed", linewidths=1.5)
        if show_Cs:
            ax.plot(0, self.interface_concentration, "o", color="#cd423b")
        ax.set_ybound(y_lims["ymin"], y_lims["ymax"])
        ax.set_xbound(-1 * solution_region_width, solution_region_width)

        ax.set_xlabel("Position (µm)")
        ax.set_ylabel("Coion Concentration (M)")

        return {
            "fig": fig,
            "ax": ax,
            "left_df": left_concentration_profile,
            "right_df": right_concentration_profile,
        }


class BPMSystem:
    def __init__(
        self,
        CEM,
        AEM,
        left_solution_concentration=0,
        right_solution_concentration=0.5,
    ):
        self.CEM = CEM
        self.AEM = AEM
        self.left_solution_concentration = left_solution_concentration * u["M"]
        self.right_solution_concentration = right_solution_concentration * u["M"]

        self.CEM_right = BPMModel(
            self.left_solution_concentration,
            self.right_solution_concentration,
            self.AEM,
            self.CEM,
        )
        self.CEM_left = BPMModel(
            self.left_solution_concentration,
            self.right_solution_concentration,
            self.CEM,
            self.AEM,
        )

        if self.left_solution_concentration > self.right_solution_concentration:
            self.CEM_high = self.CEM_left
            self.AEM_high = self.CEM_right
            self.CEM_low = self.CEM_right
            self.AEM_low = self.CEM_left
        else:
            self.CEM_high = self.CEM_right
            self.AEM_high = self.CEM_left
            self.CEM_low = self.CEM_left
            self.AEM_low = self.CEM_right

    def get_flux_differential(self):
        flux_diff = (
            200
            * (self.AEM_high.flux - self.CEM_high.flux)
            / (self.AEM_high.flux + self.CEM_high.flux)
        )
        return flux_diff.magnitude

    def display_membrane_parameters(self):
        output_string = self.CEM.__repr__() + "\n" + self.AEM.__repr__()
        print(output_string)

    @np.vectorize
    def calculate_flux_diff_with_fixed_charge(
        self, CEM_fixed_charge_concentration, AEM_fixed_charge_concentration, **kwargs
    ):
        """Return the flux differential for a given AEM and CEM fixed charge.

        Provide the CEM fixed charge concentration (in M) and AEM fixed charge
        concentration (in M) and return the flux differential. Note the fixed
        charges must be provided in molarity, and not with pint units.
        """
        self.CEM.update_parameters(
            fixed_charge_concentration=CEM_fixed_charge_concentration * u["M"]
        )
        self.AEM.update_parameters(
            fixed_charge_concentration=AEM_fixed_charge_concentration * u["M"]
        )
        self.CEM_high._calculate_bpm_flux()
        self.AEM_high._calculate_bpm_flux()
        return self.get_flux_differential()

    def calculate_flux_diff_with_diffusion_coefficients(
        self,
        cation_diffusion_coefficient,
        anion_diffusion_coefficient,
        membrane,
    ):
        """Return the flux differential for a given AEM and CEM for a new D.

        Set a new diffusion coefficient for either the AEM or CEM (as specified
        by the membrane) and return the new flux differential.

        The diffusion coefficients must be provided without pint units, with the
        magnitude in cm²/s.
        """
        membrane_object = {"CEM": self.CEM, "AEM": self.AEM}[membrane]
        membrane_object.update_parameters(
            cation_diffusion_coefficient=cation_diffusion_coefficient * u["cm²/s"],
            anion_diffusion_coefficient=anion_diffusion_coefficient * u["cm²/s"],
        )
        self.CEM_high._calculate_bpm_flux()
        self.AEM_high._calculate_bpm_flux()
        return self.get_flux_differential()

    @np.vectorize
    def calculate_flux_diff_with_D(
        self,
        cation_diffusion_coefficient,
        anion_diffusion_coefficient,
        membrane,
        **kwargs,
    ):
        """Return the flux differential for a given AEM and CEM for a new D.
        (vectorized version)

        Set a new diffusion coefficient for either the AEM or CEM (as specified
        by the membrane) and return the new flux differential.

        The diffusion coefficients must be provided without pint units, with the
        magnitude in cm²/s.
        """
        return self.calculate_flux_diff_with_diffusion_coefficients(
            cation_diffusion_coefficient, anion_diffusion_coefficient, membrane
        )

    @np.vectorize
    def calculate_flux_diff_with_D_counter_co(
        self,
        counterion_diffusion_coefficient,
        coion_diffusion_coefficient,
        membrane,
        **kwargs,
    ):
        """Same as calculate_flux_diff_with_D, except lets
        you feed the counterion diffusion coefficients as the X-values, and the
        coion diffusion coefficients as the Y-values
        """
        cation_diffusion_coefficient = {
            "CEM": counterion_diffusion_coefficient,
            "AEM": coion_diffusion_coefficient,
        }[membrane]
        anion_diffusion_coefficient = {
            "AEM": counterion_diffusion_coefficient,
            "CEM": coion_diffusion_coefficient,
        }[membrane]
        return self.calculate_flux_diff_with_diffusion_coefficients(
            cation_diffusion_coefficient, anion_diffusion_coefficient, membrane
        )

    def update_membrane_parameters(self, membrane_to_change, **kwargs):
        if membrane_to_change == "CEM":
            membrane = self.CEM
        elif membrane_to_change == "AEM":
            membrane = self.AEM
        membrane.update_parameters(**kwargs)
        self.AEM_high.compute_bpm_flux()
        self.CEM_high.compute_bpm_flux()

    def update_CEM_parameters(self, **kwargs):
        self.update_membrane_parameters("CEM", **kwargs)

    def update_AEM_parameters(self, **kwargs):
        self.update_membrane_parameters("AEM", **kwargs)

    # Generate plots
    def generate_differential_plot(
        self,
        z_function,
        x_range,
        y_range,
        v_range=None,
        n_points=20,
        mesh_type="linear",
        contourf_args={},
        colorbar_args={},
        auto_vminmax=True,
        text_location="right",
        **kwargs,
    ):
        function_dict = {
            "fixed_charge_concentration": self.calculate_flux_diff_with_fixed_charge,
            "ion_diffusion_coefficients": self.calculate_flux_diff_with_D,
            "ion_diffusion_coefficients_counter_co": self.calculate_flux_diff_with_D_counter_co,
        }

        if mesh_type == "linear":
            point_generator = np.linspace
        elif mesh_type == "log":
            point_generator = np.geomspace

        # Preserve previous membrane settings
        preserved_AEM_parameters = self.AEM.parameters.copy()
        preserved_CEM_parameters = self.CEM.parameters.copy()

        X_range = point_generator(x_range[0], x_range[1], n_points)
        Y_range = point_generator(y_range[0], y_range[1], n_points)
        XX, YY = np.meshgrid(X_range, Y_range)
        ZZ = function_dict[z_function](self, XX, YY, **kwargs)

        # Restore previous membrane settings
        self.AEM.update_parameters(**preserved_AEM_parameters)
        self.CEM.update_parameters(**preserved_CEM_parameters)

        if v_range:
            vmin = v_range[0]
            vmax = v_range[1]
        else:
            vmax = np.max(np.abs(ZZ))
            vmin = -1 * vmax

        fig, ax = plt.subplots(figsize=(3.534, 2.953))
        default_contourf_args = {
            "levels": 10,
            "cmap": "coolwarm_r",
        }
        if auto_vminmax:
            default_contourf_args["vmin"] = vmin
            default_contourf_args["vmax"] = vmax
        contourf_args = default_contourf_args | contourf_args

        im = ax.contourf(XX, YY, ZZ, **contourf_args)
        colorbar = fig.colorbar(im, ax=ax, extend="both", **colorbar_args)

        # Set Labels:
        if text_location == "right":
            X_text_location = 0.92
            Y_text_location = 0.88
        elif text_location == "below":
            X_text_location = 0.14
            Y_text_location = -0.04

        if z_function == "fixed_charge_concentration":
            ax.set_xlabel("CEM Fixed Charge Concentration")
            ax.set_ylabel("AEM Fixed Charge Concentration")
            if not text_location == "off":
                fig.text(
                    X_text_location,
                    Y_text_location,
                    f"CEM:\n$D_+$ = {self.CEM.cation_diffusion_coefficient:.2e~P}"
                    + f"\n$D_-$ = {self.CEM.anion_diffusion_coefficient:.2e~P}"
                    + f"\n\nAEM:\n$D_+$ = {self.AEM.cation_diffusion_coefficient:.2e~P}"
                    + f"\n$D_-$ = {self.AEM.anion_diffusion_coefficient:.2e~P}",
                    va="top",
                )

        def generate_base_ion_diffusion_coefficient_text(fig, ax):
            CEM_label = f"CEM:\n$X_m$ = {self.CEM.fixed_charge_concentration:.2e~P}"
            AEM_label = f"\n\nAEM:\n$X_m$ = {self.AEM.fixed_charge_concentration:.2e~P}"
            if kwargs["membrane"] == "AEM":
                CEM_label = (
                    CEM_label
                    + f"\n$D_+$ = {self.CEM.cation_diffusion_coefficient:.2e~P}"
                    + f"\n$D_-$ = {self.CEM.anion_diffusion_coefficient:.2e~P}"
                )
            else:
                AEM_label = (
                    AEM_label
                    + f"\n$D_+$ = {self.AEM.cation_diffusion_coefficient:.2e~P}"
                    + f"\n$D_-$ = {self.AEM.anion_diffusion_coefficient:.2e~P}"
                )
            if not text_location == "off":
                fig.text(
                    X_text_location,
                    Y_text_location,
                    CEM_label + AEM_label,
                    va="top",
                )

        if z_function == "ion_diffusion_coefficients":
            membrane_type = kwargs["membrane"]
            generate_base_ion_diffusion_coefficient_text(fig, ax)
            ax.set_xlabel(f"Cation Diffusion Coefficient in {membrane_type} (cm²/s)")
            ax.set_ylabel(f"Anion Diffusion Coefficient in {membrane_type} (cm²/s)")

        if z_function == "ion_diffusion_coefficients_counter_co":
            membrane_type = kwargs["membrane"]
            generate_base_ion_diffusion_coefficient_text(fig, ax)
            ax.set_xlabel(
                f"Counterion Diffusion Coefficient in {membrane_type} (cm²/s)", size=9
            )
            ax.set_ylabel(
                f"Coion Diffusion Coefficient in {membrane_type} (cm²/s)", size=9
            )

        if mesh_type == "log":
            ax.set_xscale("log")
            ax.set_yscale("log")

        data = [XX, YY, ZZ, colorbar]
        return (fig, ax, data)
