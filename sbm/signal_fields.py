# -*- encoding: utf-8 -*-

import numpy as np
import copy
import healpy as hp
from .convolver import Convolver
import sympy as sp
from typing import List


class Field:
    """Class to store the field data of detectors"""

    def __init__(self, field: np.ndarray, spin_n: float, spin_m: float):
        """Initialize the class with field and spin data

        Args:
            field (`np.ndarray`): field data of the detector

            spin_n (`float`): spin moment of the crossing angle

            spin_m (`float`): spin moment of the HWP angle
        """
        if all(isinstance(x, float) for x in field):
            self.field = field + 1j * np.zeros(len(field))
        else:
            self.field = field
        self.npix = len(field)
        self.nside = hp.npix2nside(self.npix)
        self.spin_n = spin_n
        self.spin_m = spin_m

    def conj(self):
        """Get the complex conjugate of the field"""
        return Field(self.field.conj(), -self.spin_n, -self.spin_m)

    def __add__(self, other):
        """Add the field of two detectors

        Args:
            other (:class:`.Field`): field of the other detector

        Returns:
            result (:class:`.Field`): sum of the fields of the two detectors
        """
        if not isinstance(other, Field):
            return NotImplemented
        result = copy.deepcopy(self)
        result.field += other.field
        return result


class SignalFields:
    """Class to store the signal fields data of detectors"""

    def __init__(self, *fields: Field):
        """Initialize the class with field data

        Args:
            fields (:class:`.Field`): field (map) data of the signal
        """
        # Expand if fields are passed in a list
        if len(fields) == 1 and isinstance(fields[0], list):
            fields = fields[0]
        self.fields = sorted(fields, key=lambda field: (field.spin_n, field.spin_m))
        self.nside = self.fields[0].nside
        self.npix = self.fields[0].npix
        self.spins_n = np.array([field.spin_n for field in self.fields])
        self.spins_m = np.array([field.spin_m for field in self.fields])
        self.field_name = None
        self.coupled_fields = None
        self.spin_n_basis = []
        self.spin_m_basis = []

    def get_field(self, spin_n: float, spin_m: float):
        """Get the field of the given spin moment

        Args:
            spin_n (`float`): spin moment for which the field is to be obtained

            spin_m (`float`): spin moment for which the field is to be obtained

        Returns:
            field (`np.ndarray`): field of the detector for the given spin moment
        """
        for field in self.fields:
            if field.spin_n == spin_n and field.spin_m == spin_m:
                return field.field
        assert False, f"Field with spin_n={spin_n} and spin_m={spin_m} not found"

    def extract_iqu(self):
        """Extract I, Q and U maps from the signal fields"""
        try:
            stokes_i = self.get_field(0, 0)
        except AssertionError:
            stokes_i = np.zeros(self.npix)
        if np.all(self.spin_m_basis == 0):
            # without HWP
            stokes_q = self.get_field(2, 0).real
            stokes_u = self.get_field(2, 0).imag
        else:
            # with HWP
            stokes_q = self.get_field(-2, 4).real
            stokes_u = self.get_field(-2, 4).imag
        return np.array([stokes_i, stokes_q, stokes_u])

    def __add__(self, other):
        """Add the signal fields

        Args:
            other (:class:`.SignalFields`): signal fields

        Returns:
            result (:class:`.SignalFields`): sum of the signal fields
        """
        if not isinstance(other, SignalFields):
            return NotImplemented
        result = copy.deepcopy(self)
        for i in range(len(self.spins_n)):
            result.fields[i].field += other.fields[i].field
        return result

    def get_coupled_field(
        self, scan_field, spin_n_out: float, spin_m_out: float, output_all=False
    ):
        """Multiply the scan fields and signal fields to get the detected fields by
        given cross-linking

        Args:
            scan_field (:class:`.ScanFields`): scan fields instance

            spin_n_out (`float`): spin_n of the output field

            spin_m_out (`float`): spin_m of the output field

        Returns:
            results (`np.ndarray`): detected fields by the given cross-linking
        """
        results = []
        h_name = []
        S_name = []
        delta_nm = []
        nm = []
        for i in range(len(self.spins_n)):
            n = self.spins_n[i]
            m = self.spins_m[i]
            delta_n = spin_n_out - n
            delta_m = spin_m_out - m
            hS = scan_field.get_xlink(delta_n, delta_m) * self.get_field(n, m)
            results.append(hS)
            h_name.append(rf"${{}}_{{{delta_n},{delta_m}}}\tilde{{h}}$")
            S_name.append(rf"${{}}_{{{n},{m}}}\tilde{{S}}$")
            delta_nm.append((delta_n, delta_m))
            nm.append((n, m))
        results = np.array(results)
        if output_all is True:
            return {
                "results": results,
                "h_name": h_name,
                "S_name": S_name,
                "delta_nm": delta_nm,
                "nm": nm,
            }
        else:
            return Field(results.sum(0), spin_n_out, spin_m_out)

    def build_linear_system(self, fields: list):
        """Build the information to solve the linear system of map-making
        This method has to be called before :meth:`.ScanFields.map_make()` method.

        Args:
            fields (`list`): list of coupled fields, its element must be :class:`.Field` instance
        """
        coupled_fields = []
        spin_n_basis = []
        spin_m_basis = []
        for field in fields:
            assert isinstance(
                field, Field
            ), "element of `fields` must be `Field` instance"
            if field.spin_n == 0 and field.spin_m == 0:
                coupled_fields.append(field.field)
            else:
                coupled_fields.append(field.field / 2.0)
            spin_n_basis.append(field.spin_n)
            spin_m_basis.append(field.spin_m)
        spin_n_basis = np.array(spin_n_basis)
        spin_m_basis = np.array(spin_m_basis)
        self.coupled_fields = np.array(coupled_fields)
        if all(spin_m_basis == 0):
            self.spin_n_basis = spin_n_basis
            self.spin_m_basis = spin_m_basis
        else:
            # HWP on, the sign of spin basis is flipped
            self.spin_n_basis = -spin_n_basis
            self.spin_m_basis = -spin_m_basis
        model_vector = sp.zeros(len(self.spin_n_basis), 1)
        solved_vector = sp.zeros(len(self.spin_n_basis), 1)
        for i in range(len(spin_n_basis)):
            n = spin_n_basis[i]
            m = spin_m_basis[i]
            model_vector[i] = sp.Symbol(rf"{{}}_{{{n},{m}}}\tilde{{S^d}}")
            if all(spin_m_basis == 0):
                if n == 0:
                    stokes_element = sp.Symbol(r"{}_{0,0}\hat{{Z}}")
                elif n == 2:
                    stokes_element = sp.Symbol(r"\hat{P}")
                elif n == -2:
                    stokes_element = sp.Symbol(r"\hat{P^*}")
                else:
                    stokes_element = sp.Symbol(rf"{{}}_{{{n},{m}}}\hat{{Z}}")
            else:
                if n == 0:
                    stokes_element = sp.Symbol(r"\hat{I}")
                elif n == 2:
                    stokes_element = sp.Symbol(r"\hat{P}")
                elif n == -2:
                    stokes_element = sp.Symbol(r"\hat{P^*}")
                else:
                    stokes_element = sp.Symbol(rf"{{}}_{{{n},{m}}}\hat{{Z}}")
            solved_vector[i] = stokes_element
        self.model_vector = model_vector
        self.solved_vector = solved_vector

    @staticmethod
    def diff_gain_field(
        scan_field,
        mdim: int,
        gain_T: float,
        gain_B: float,
        temp_map: np.ndarray,
        pol_map: np.ndarray,
    ):
        """Get the differential gain field of the detector

        Args:
            scan_field (:class:`.ScanFields`): scan field instance

            mdim (`int`): dimension of the map-making liner system

            gain_T (`float`): gain of the `Top` detector

            gain_B (`float`): gain of the `Bottom` detector

            temp_map (`np.ndarray`): temperature map

            pol_map (`np.ndarray`): polarization map (i.e. Q+iU)

        Returns:
            signal_fields (:class:`.SignalFields`): differential gain field of the detector
        """
        delta_g = gain_T - gain_B
        signal_fields = SignalFields(
            Field(delta_g * temp_map / 2.0, spin_n=0, spin_m=0),
            Field((2.0 + gain_T + gain_B) * pol_map / 4.0, spin_n=2, spin_m=0),
            Field((2.0 + gain_T + gain_B) * pol_map.conj() / 4.0, spin_n=-2, spin_m=0),
        )
        signal_fields.field_name = "diff_gain_field"
        s_0 = signal_fields.get_coupled_field(scan_field, spin_n_out=0, spin_m_out=0)
        sp2 = signal_fields.get_coupled_field(scan_field, spin_n_out=2, spin_m_out=0)
        if mdim == 2:
            fields = [sp2, sp2.conj()]
        elif mdim == 3:
            fields = [s_0, sp2, sp2.conj()]
        else:
            raise ValueError("mdim is 2 and 3 only supported")
        signal_fields.build_linear_system(fields)
        return signal_fields

    @staticmethod
    def diff_pointing_field(
        scan_field,
        mdim: int,
        rho_T: float,
        rho_B: float,
        chi_T: float,
        chi_B: float,
        pol_map: np.ndarray,
        eth_temp_map: np.ndarray,
        eth_pol_map: np.ndarray,
        o_eth_pol_map: np.ndarray,
    ):
        """Get the differential pointing field of the detector

        Args:
            scan_field (:class:`.ScanFields`): scan field instance

            mdim (`int`): dimension of the map-making liner system

            rho_T (`float`): magnitude of pointing offset of the `Top` detector in radian

            rho_B (`float`): magnitude of pointing offset of the `Bottom` detector in radian

            chi_T (`float`): direction of the pointing offset of the `Top` detector in radian

            chi_B (`float`): direction of the pointing offset of the `Bottom` detector in radian

            pol_map (`np.ndarray`): polarization map (i.e. Q+iU)

            eth_temp_map (`np.ndarray`): spin up gradient of the temperature map

            eth_pol_map (`np.ndarray`): spin up gradient of the polarization map

            o_eth_pol_map (`np.ndarray`): spin down gradient of the polarization map
        """
        zeta = rho_T * np.exp(1j * chi_T) - 1j * rho_B * np.exp(1j * chi_B)
        o_zeta = rho_T * np.exp(1j * chi_T) + 1j * rho_B * np.exp(
            1j * chi_B
        )  # \overline{\zeta}

        spin_1_field = Field(
            -1.0 / 4.0 * (zeta * eth_temp_map + o_zeta.conj() * o_eth_pol_map),
            spin_n=1,
            spin_m=0,
        )
        spin_m1_field = spin_1_field.conj()
        spin_2_field = Field(pol_map / 2.0, spin_n=2, spin_m=0)
        spin_m2_field = spin_2_field.conj()
        spin_3_field = Field(-1.0 / 4.0 * o_zeta * eth_pol_map, spin_n=3, spin_m=0)
        spin_m3_field = spin_3_field.conj()

        signal_fields = SignalFields(
            spin_1_field,
            spin_m1_field,
            spin_2_field,
            spin_m2_field,
            spin_3_field,
            spin_m3_field,
        )
        signal_fields.field_name = "diff_pointing_field"
        sp2 = signal_fields.get_coupled_field(scan_field, spin_n_out=2, spin_m_out=0)
        if mdim == 2:
            fields = [sp2, sp2.conj()]
        elif mdim == 4:  # T grad. mitigation
            sp1 = signal_fields.get_coupled_field(
                scan_field, spin_n_out=1, spin_m_out=0
            )
            fields = [sp1, sp1.conj(), sp2, sp2.conj()]
        elif mdim == 6:  # Temp. and Pol. grad. mitigation
            sp1 = signal_fields.get_coupled_field(
                scan_field, spin_n_out=1, spin_m_out=0
            )
            sp3 = signal_fields.get_coupled_field(
                scan_field, spin_n_out=3, spin_m_out=0
            )
            fields = [sp1, sp1.conj(), sp2, sp2.conj(), sp3, sp3.conj()]
        else:
            raise ValueError("mdim is 2,4 and 6 only supported")
        signal_fields.build_linear_system(fields)
        return signal_fields

    @staticmethod
    def bandpass_mismatch_field(
        scan_field,
        mdim: int,
        pol_map: np.ndarray,
        gamma_T_list: List[float],
        gamma_B_list: List[float],
        components: List[np.ndarray],
    ):
        """Get the bandpass mismatch field of the detector
        The formalism is based on the paper by Duc Thuong Hoang et al., 2017, JCAP,
        DOI: 10.1088/1475-7516/2017/12/015, Sec. 3.2

        Args:
            scan_field (:class:`.ScanFields`): scan field instance

            mdim (`int`): dimension of the map-making liner system

            pol_map (`np.ndarray`): polarization map (i.e. Q+iU)

            gamma_T_list (`List[float]`): list of coefficient of the sky signal which corresponds to the bandpass mismatch of the `Bottom` detector

            gamma_B_list (`List[float]`): list of coefficient of the sky signal which corresponds to the bandpass mismatch of the `Top` detector

            components (`List[np.ndarray]`): list of the components (e.g. synchrotron and dust) of the temperature sky map
        """
        assert len(gamma_T_list) == len(
            gamma_B_list
        ), "gamma_T_list and gamma_B_list must have the same length"
        assert len(gamma_T_list) == len(
            components
        ), "gamma_T_list, gamma_B_list and components must have the same length"

        bpm_comp = np.zeros(scan_field.npix)
        for i in range(len(components)):
            bpm_comp += 1.0 / 2.0 * (gamma_T_list[i] - gamma_B_list[i]) * components[i]

        signal_fields = SignalFields(
            Field(bpm_comp, spin_n=0, spin_m=0),
            Field(pol_map / 2.0, spin_n=2, spin_m=0),
            Field(pol_map.conj() / 2.0, spin_n=-2, spin_m=0),
        )
        signal_fields.field_name = "bandpass_mismatch_field"
        s_0 = signal_fields.get_coupled_field(scan_field, spin_n_out=0, spin_m_out=0)
        sp2 = signal_fields.get_coupled_field(scan_field, spin_n_out=2, spin_m_out=0)
        if mdim == 2:
            fields = [sp2, sp2.conj()]
        elif mdim == 3:
            fields = [s_0, sp2, sp2.conj()]
        else:
            raise ValueError("mdim is 2 and 3 only supported")
        signal_fields.build_linear_system(fields)
        return signal_fields

    @staticmethod
    def hwp_ip_field(
        scan_field,
        mdim: int,
        epsilon: float,
        phi_qi: float,
        temp_map: np.ndarray,
    ):
        """Get the HWP instrumental polarization field of the detector

        Args:
            scan_field (:class:`.ScanFields`): scan field instance

            mdim (`int`): dimension of the map-making liner system

            epsilon (`float`): amplitude of the HWP Meuller matrix element varying at 4f

            phi_qi (`float`): phase shift in the HWP

            temp_map (`np.ndarray`): temperature map
        """
        spin_4m4_field = Field(
            epsilon / 2.0 * np.exp(-1j * phi_qi) * temp_map, spin_n=4, spin_m=-4
        )
        signal_fields = SignalFields(
            spin_4m4_field,
            spin_4m4_field.conj(),
        )
        signal_fields.field_name = "hwp_ip_field"
        s_00 = signal_fields.get_coupled_field(scan_field, spin_n_out=0, spin_m_out=0)
        sp2m4 = signal_fields.get_coupled_field(scan_field, spin_n_out=2, spin_m_out=-4)
        if mdim == 3:
            fields = [s_00, sp2m4, sp2m4.conj()]
        elif mdim == 5:
            sp4m4 = signal_fields.get_coupled_field(
                scan_field, spin_n_out=4, spin_m_out=-4
            )
            fields = [s_00, sp2m4, sp2m4.conj(), sp4m4, sp4m4.conj()]
        else:
            raise ValueError("mdim is 3 and 5 only supported")
        signal_fields.build_linear_system(fields)
        return signal_fields

    @staticmethod
    def abs_pointing_field(
        scan_field,
        mdim: int,
        rho: float,
        chi: float,
        temp_map: np.ndarray,
        pol_map: np.ndarray,
        eth_temp_map: np.ndarray,
        eth_pol_map: np.ndarray,
        o_eth_pol_map: np.ndarray,
    ):
        """Get the absolute pointing field of the detector

        Args:
            scan_field (:class:`.ScanFields`): scan field instance

            mdim (`int`): dimension of the map-making liner system

            rho (`float`): magnitude of pointing offset in radian

            chi (`float`): direction of the pointing offset in radian

            temp_map (`np.ndarray`): temperature map

            pol_map (`np.ndarray`): polarization map (i.e. Q+iU)

            eth_temp_map (`np.ndarray`): spin up gradient of the temperature map

            eth_pol_map (`np.ndarray`): spin up gradient of the polarization map

            o_eth_pol_map (`np.ndarray`): spin down gradient of the polarization map
        """
        spin_00_field = Field(temp_map, spin_n=0, spin_m=0)
        spin_p2m4_field = Field(pol_map / 2.0, spin_n=2, spin_m=-4)
        spin_m2p4_field = spin_p2m4_field.conj()
        spin_p10_field = Field(
            -rho / 2.0 * np.exp(1j * chi) * eth_temp_map, spin_n=1, spin_m=0
        )
        spin_m10_field = spin_p10_field.conj()
        spin_p1m4_field = Field(
            -rho / 4.0 * np.exp(-1j * chi) * o_eth_pol_map, spin_n=1, spin_m=-4
        )
        spin_m1p4_field = spin_p1m4_field.conj()
        spin_p3m4_field = Field(
            -rho / 4.0 * np.exp(1j * chi) * eth_pol_map, spin_n=3, spin_m=-4
        )
        spin_m3p4_field = spin_p3m4_field.conj()
        signal_fields = SignalFields(
            spin_00_field,
            spin_p10_field,
            spin_m10_field,
            spin_p2m4_field,
            spin_m2p4_field,
            spin_p3m4_field,
            spin_m3p4_field,
            spin_p1m4_field,
            spin_m1p4_field,
        )
        signal_fields.field_name = "abs_pointing_field"
        s_00 = signal_fields.get_coupled_field(scan_field, spin_n_out=0, spin_m_out=0)
        sp2m4 = signal_fields.get_coupled_field(scan_field, spin_n_out=2, spin_m_out=-4)
        if mdim == 3:
            fields = [s_00, sp2m4, sp2m4.conj()]
        elif mdim == 5:
            sp10 = signal_fields.get_coupled_field(
                scan_field, spin_n_out=1, spin_m_out=0
            )
            fields = [s_00, sp10, sp10.conj(), sp2m4, sp2m4.conj()]
        elif mdim == 9:
            sp10 = signal_fields.get_coupled_field(
                scan_field, spin_n_out=1, spin_m_out=0
            )
            sp3m4 = signal_fields.get_coupled_field(
                scan_field, spin_n_out=3, spin_m_out=-4
            )
            sp1m4 = signal_fields.get_coupled_field(
                scan_field, spin_n_out=1, spin_m_out=-4
            )
            fields = [
                s_00,
                sp10,
                sp10.conj(),
                sp2m4,
                sp2m4.conj(),
                sp3m4,
                sp3m4.conj(),
                sp1m4,
                sp1m4.conj(),
            ]
        else:
            raise ValueError("mdim is 3,5 and 9 only supported")
        signal_fields.build_linear_system(fields)
        return signal_fields

    @staticmethod
    def circular_pointing_field(
        scan_field,
        mdim: int,
        rho: float,
        chi: float,
        temp_map: np.ndarray,
        pol_map: np.ndarray,
        eth_temp_map: np.ndarray,
        eth_pol_map: np.ndarray,
        o_eth_pol_map: np.ndarray,
    ):
        """Get the absolute pointing field of the detector

        Args:
            scan_field (:class:`.ScanFields`): scan field instance

            mdim (`int`): dimension of the map-making liner system

            rho (`float`): magnitude of pointing offset in radian

            chi (`float`): direction of the pointing offset in radian

            temp_map (`np.ndarray`): temperature map

            pol_map (`np.ndarray`): polarization map (i.e. Q+iU)

            eth_temp_map (`np.ndarray`): spin up gradient of the temperature map

            eth_pol_map (`np.ndarray`): spin up gradient of the polarization map

            o_eth_pol_map (`np.ndarray`): spin down gradient of the polarization map
        """
        spin_00_field = Field(temp_map, spin_n=0, spin_m=0)
        spin_p2m4_field = Field(pol_map / 2.0, spin_n=2, spin_m=-4)
        spin_m2p4_field = spin_p2m4_field.conj()
        spin_p1p1_field = Field(
            -rho / 2.0 * np.exp(1j * chi) * eth_temp_map, spin_n=1, spin_m=1
        )
        spin_m1m1_field = spin_p1p1_field.conj()
        spin_p1m5_field = Field(
            -rho / 4.0 * np.exp(-1j * chi) * o_eth_pol_map, spin_n=1, spin_m=-5
        )
        spin_m1p5_field = spin_p1m5_field.conj()
        spin_p3m3_field = Field(
            -rho / 4.0 * np.exp(1j * chi) * eth_pol_map, spin_n=3, spin_m=-3
        )
        spin_m3p3_field = spin_p3m3_field.conj()
        signal_fields = SignalFields(
            spin_00_field,
            spin_p2m4_field,
            spin_m2p4_field,
            spin_p1p1_field,
            spin_m1m1_field,
            spin_p3m3_field,
            spin_m3p3_field,
            spin_p1m5_field,
            spin_m1p5_field,
        )
        signal_fields.field_name = "hwp_wedge_field"
        s_00 = signal_fields.get_coupled_field(scan_field, spin_n_out=0, spin_m_out=0)
        sp2m4 = signal_fields.get_coupled_field(scan_field, spin_n_out=2, spin_m_out=-4)
        if mdim == 3:
            fields = np.array([s_00, sp2m4, sp2m4.conj()])
        elif mdim == 5:
            sp1p1 = signal_fields.get_coupled_field(
                scan_field, spin_n_out=1, spin_m_out=1
            )
            fields = np.array([s_00, sp1p1, sp1p1.conj(), sp2m4, sp2m4.conj()])
        elif mdim == 9:
            sp1p1 = signal_fields.get_coupled_field(
                scan_field, spin_n_out=1, spin_m_out=1
            )
            sp1m5 = signal_fields.get_coupled_field(
                scan_field, spin_n_out=1, spin_m_out=-5
            )
            sp3m3 = signal_fields.get_coupled_field(
                scan_field, spin_n_out=3, spin_m_out=-3
            )
            fields = np.array(
                [
                    s_00,
                    sp1p1,
                    sp1p1.conj(),
                    sp2m4,
                    sp2m4.conj(),
                    sp3m3,
                    sp3m3.conj(),
                    sp1m5,
                    sp1m5.conj(),
                ]
            )
        else:
            raise ValueError("mdim is 3,5 and 9 only supported")
        signal_fields.build_linear_system(fields)
        return signal_fields

    @staticmethod
    def elliptical_beam_field(
        scan_field,
        mdim: int,
        alm: np.ndarray,
        blm: np.ndarray,
        use_hwp=False,
    ):
        """Get the elliptical beam convolved field

        Args:
            scan_field (:class:`.ScanFields`): scan field instance

            mdim (`int`): dimension of the map-making liner system

            alm (`np.ndarray`): spherical harmonic expansion coefficients of sky

            blm (`np.ndarray`): spherical harmonic expansion coefficients of beam

            use_hwp (`bool`): whether the observation uses HWP or not

        Returns:
            signal_fields (:class:`.SignalFields`): elliptical beam convolution field of the detector
        """
        alm_conv = Convolver(
            alm=alm, nside=scan_field.nside, spin_k=[0, 2, 4], use_hwp=use_hwp
        )
        blm_conv = Convolver(
            alm=blm, nside=scan_field.nside, spin_k=[0, 2, 4], use_hwp=use_hwp
        )
        all_maps = alm_conv * blm_conv

        signal_fields = SignalFields(
            Field(all_maps[0][0] + all_maps[0][1], spin_n=0, spin_m=0),
            Field((all_maps[1][0] + all_maps[1][1]) / 2, spin_n=2, spin_m=0),
            Field((all_maps[1][0] + all_maps[1][1]).conj() / 2, spin_n=-2, spin_m=0),
            Field((all_maps[2][0] + all_maps[2][1]) / 2, spin_n=4, spin_m=0),
            Field((all_maps[2][0] + all_maps[2][1]).conj() / 2, spin_n=-4, spin_m=0),
        )

        s_0 = signal_fields.get_coupled_field(scan_field, spin_n_out=0, spin_m_out=0)
        sp2 = signal_fields.get_coupled_field(scan_field, spin_n_out=2, spin_m_out=0)
        signal_fields.field_name = "elliptical_beam_field"
        if mdim == 2:
            fields = [sp2, sp2.conj()]
        elif mdim == 3:
            fields = [s_0, sp2, sp2.conj()]
        elif mdim == 5:
            sp4 = signal_fields.get_coupled_field(
                scan_field, spin_n_out=4, spin_m_out=0
            )
            fields = [s_0, sp2, sp2.conj(), sp4, sp4.conj()]
        else:
            raise ValueError("mdim is 2,3 and 5 only supported")
        signal_fields.build_linear_system(fields)
        return signal_fields
