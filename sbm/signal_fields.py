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


# HWP angle conventions for the HWP-synchronous systematics (wedge, HWP-IP).
#
# All SBM fields are evaluated against the same orientation functions
# h_{n,m} = <exp(-i(n psi + m phi))>, with phi = omega_HWP t, and the polarization
# is placed at spin (2,-4), i.e. the TOD modulation is cos(4 phi - 2 psi). Which
# physical angles ``psi`` and ``phi`` stand for fixes the spins of every signal that
# depends on the HWP orientation, so the wedge and the HWP-IP must be written in the
# same convention.
#
# Physically, let psi_0 be the crossing angle and a the HWP angle in the focal plane,
# both counted in the sense of SBM's crossing angle without HWP (the sense in which
# the polarization enters as P exp(2i psi_0) and the absolute pointing offset points
# along psi_0 + chi). The HWP modulates the polarization as P exp(i(2 psi_0 + 4a)),
# the wedge, fixed to the HWP, deflects the beam along psi_0 + a + chi, and the 4f
# HWP-IP depends on 4a = 2 (2 psi_0 + 4a) - 4 (psi_0 + a) alone. Offset directions
# and ``chi`` are counted in this sense in every convention. The HWP-IP is
# epsilon cos(4 phi_fp + phi_qi) as in Patanchon et al. (2024), with phi_fp = -a the
# focal-plane HWP angle counted in the standard sense (that of litebird_sim's HWP
# angle); its route exp(-i phi_qi) exp(4ia) is at spin IP = 2 pol - 4 wedge_T,
# whatever the angle variables.
#
# "sky"        phi is the HWP angle with respect to the sky reference frame
#              (Patanchon et al. 2024, rho = omega t) and the angles are counted in
#              the opposite sense, psi = -psi_0, phi = -(psi_0 + a). The HWP angle in
#              the focal plane is phi - psi = -a, the wedge offset points along
#              psi_0 + a + chi = -phi + chi (spin (0,-1)), and the 4f HWP-IP (a
#              function of the focal-plane HWP angle) sits at spin (4,-4).
# "telescope"  phi is the HWP rotation angle with respect to the telescope, on which
#              the HWP is mounted; cos(4 phi - 2 psi) then implies that phi is
#              counted in the opposite sense to psi: psi = psi_0, phi = -a. This is
#              the convention of the h-maps of litebird_sim and Falcons.jl read by
#              ScanFields.load_lbs_h_maps (phi is litebird_sim's HWP angle).
#              The wedge offset points along psi - phi + chi (spin (1,-1)), the HWP
#              angle in the focal plane is phi = -a, and the HWP-IP sits at spin
#              (0,+-4), its route exp(-i phi_qi) at (0,-4).
# "legacy"     previous SBM behaviour: HWP-IP as in "sky", wedge offset along
#              psi + phi + chi. The two are not mutually consistent (the identity
#              IP = 2 pol - 4 wedge_T between spin vectors fails); kept only to
#              reproduce earlier results.
HWP_ANGLE_FRAMES = ("legacy", "sky", "telescope")


def _check_hwp_angle_frame(hwp_angle_frame: str) -> None:
    if hwp_angle_frame not in HWP_ANGLE_FRAMES:
        raise ValueError(
            f"hwp_angle_frame must be one of {HWP_ANGLE_FRAMES}, got {hwp_angle_frame!r}"
        )


def wedge_offset_direction_spin(hwp_angle_frame: str = "legacy") -> tuple:
    """Spin (dn, dm) of exp(i alpha) for the wedge offset direction alpha = dn psi + dm phi.

    ``"legacy"``: psi + phi, ``"sky"``: -phi, ``"telescope"``: psi - phi (see the notes
    above; the direction is counted like SBM's crossing angle without HWP, as for the
    absolute pointing offset). The first-order temperature-gradient term of the wedge
    sits at this spin.
    """
    _check_hwp_angle_frame(hwp_angle_frame)
    return {"legacy": (1, 1), "sky": (0, -1), "telescope": (1, -1)}[hwp_angle_frame]


def hwp_ip_route_spins(hwp_angle_frame: str = "legacy") -> tuple:
    """Spins of the two HWP-IP routes (route a carries exp(-i phi_qi), route b its conjugate).

    The 4f instrumental polarization ``epsilon cos(4 phi_fp + phi_qi)`` depends only on
    the focal-plane HWP angle ``phi_fp``, counted in the standard sense as in Patanchon
    et al. (2024): phi - psi (``"sky"``/``"legacy"``) or phi (``"telescope"``). Route a,
    ``exp(-i phi_qi) exp(-4i phi_fp)``, sits at (4,-4) or (0,-4).
    """
    _check_hwp_angle_frame(hwp_angle_frame)
    if hwp_angle_frame == "telescope":
        return (0, -4), (0, 4)
    return (4, -4), (-4, 4)


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
                if n == 0 and m == 0:
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

            eth_pol_map (`np.ndarray`): spin up gradient of the polarization map, the covariant
            derivative of :func:`.spin_derivatives`

            o_eth_pol_map (`np.ndarray`): spin down gradient of the polarization map, the covariant
            derivative of :func:`.spin_derivatives`
        """
        zeta = rho_T * np.exp(1j * chi_T) - 1j * rho_B * np.exp(1j * chi_B)
        o_zeta = rho_T * np.exp(1j * chi_T) + 1j * rho_B * np.exp(
            1j * chi_B
        )  # \overline{\zeta}

        # Pair difference d = (d_T - d_B)/2: the temperature gradient enters with the
        # difference of the offsets (zeta) and weight 1/4, while the polarization
        # gradient enters with their sum (o_zeta) and weight 1/8, because the
        # polarization term of each detector carries the factor 1/2 of the pair.
        spin_1_field = Field(
            -1.0 / 4.0 * zeta * eth_temp_map
            - 1.0 / 8.0 * o_zeta.conj() * o_eth_pol_map,
            spin_n=1,
            spin_m=0,
        )
        spin_m1_field = spin_1_field.conj()
        spin_2_field = Field(pol_map / 2.0, spin_n=2, spin_m=0)
        spin_m2_field = spin_2_field.conj()
        spin_3_field = Field(-1.0 / 8.0 * o_zeta * eth_pol_map, spin_n=3, spin_m=0)
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
        hwp_angle_frame: str = "legacy",
    ):
        """Get the HWP instrumental polarization field of the detector

        The 4f term ``epsilon cos(4 phi_fp + phi_qi)`` of Patanchon et al. (2024)
        depends on the focal-plane HWP angle ``phi_fp``, counted in the standard sense
        (``phi - psi`` for ``"sky"``/``"legacy"``, ``phi`` for ``"telescope"``, i.e.
        litebird_sim's HWP angle); its spin follows ``hwp_angle_frame`` (see
        :func:`hwp_ip_route_spins`): (4,-4) for ``"sky"``/``"legacy"``, (0,+-4) for
        ``"telescope"``.

        Args:
            scan_field (:class:`.ScanFields`): scan field instance

            mdim (`int`): dimension of the map-making liner system

            epsilon (`float`): amplitude of the HWP Meuller matrix element varying at 4f

            phi_qi (`float`): phase shift in the HWP, the phase of the 4f term

            temp_map (`np.ndarray`): temperature map

            hwp_angle_frame (`str`): meaning of the HWP angle ``phi`` of the scan
                fields, ``"legacy"`` (default, same as ``"sky"``), ``"sky"`` or
                ``"telescope"`` (that of :meth:`.ScanFields.load_lbs_h_maps`). Use
                the same value as for the wedge.
        """
        (na, ma), _ = hwp_ip_route_spins(hwp_angle_frame)
        route_a_field = Field(
            epsilon / 2.0 * np.exp(-1j * phi_qi) * temp_map, spin_n=na, spin_m=ma
        )
        signal_fields = SignalFields(
            route_a_field,
            route_a_field.conj(),
        )
        signal_fields.field_name = "hwp_ip_field"
        s_00 = signal_fields.get_coupled_field(scan_field, spin_n_out=0, spin_m_out=0)
        sp2m4 = signal_fields.get_coupled_field(scan_field, spin_n_out=2, spin_m_out=-4)
        if mdim == 3:
            fields = [s_00, sp2m4, sp2m4.conj()]
        elif mdim == 5:
            s_ip = signal_fields.get_coupled_field(
                scan_field, spin_n_out=na, spin_m_out=ma
            )
            fields = [s_00, sp2m4, sp2m4.conj(), s_ip, s_ip.conj()]
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

            eth_pol_map (`np.ndarray`): spin up gradient of the polarization map, the covariant
            derivative of :func:`.spin_derivatives`

            o_eth_pol_map (`np.ndarray`): spin down gradient of the polarization map, the covariant
            derivative of :func:`.spin_derivatives`
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
        hwp_angle_frame: str = "legacy",
    ):
        """Get the HWP wedge field of the detector

        The wedge is the absolute-pointing model with the offset direction
        ``psi + chi`` replaced by ``alpha + chi``, where ``alpha = dn psi + dm phi`` is
        the sky orientation of the HWP-fixed deflection
        (:func:`wedge_offset_direction_spin`): ``psi + phi`` (``"legacy"``), ``-phi``
        (``"sky"``) or ``psi - phi`` (``"telescope"``). The direction and ``chi`` are
        counted like SBM's crossing angle without HWP in every convention. The
        first-order terms sit at (dn, dm), (2+dn, -4+dm), (2-dn, -4-dm) and their
        conjugates.

        The scan fields read by :meth:`.ScanFields.load_lbs_h_maps` are in the
        ``"telescope"`` convention; the default ``"legacy"`` only reproduces earlier
        results.

        Args:
            scan_field (:class:`.ScanFields`): scan field instance

            mdim (`int`): dimension of the map-making liner system

            rho (`float`): magnitude of pointing offset in radian

            chi (`float`): direction of the pointing offset in radian, with respect
                to the HWP

            temp_map (`np.ndarray`): temperature map

            pol_map (`np.ndarray`): polarization map (i.e. Q+iU)

            eth_temp_map (`np.ndarray`): spin up gradient of the temperature map

            eth_pol_map (`np.ndarray`): spin up gradient of the polarization map, the covariant
            derivative of :func:`.spin_derivatives`

            o_eth_pol_map (`np.ndarray`): spin down gradient of the polarization map, the covariant
            derivative of :func:`.spin_derivatives`

            hwp_angle_frame (`str`): meaning of the HWP angle ``phi`` of the scan
                fields, ``"legacy"`` (default), ``"sky"`` or ``"telescope"`` (that of
                :meth:`.ScanFields.load_lbs_h_maps`). Use the same value as for
                :meth:`hwp_ip_field`.
        """
        dn, dm = wedge_offset_direction_spin(hwp_angle_frame)
        spin_00_field = Field(temp_map, spin_n=0, spin_m=0)
        spin_p2m4_field = Field(pol_map / 2.0, spin_n=2, spin_m=-4)
        spin_m2p4_field = spin_p2m4_field.conj()
        t1_field = Field(
            -rho / 2.0 * np.exp(1j * chi) * eth_temp_map, spin_n=dn, spin_m=dm
        )
        pdown_field = Field(
            -rho / 4.0 * np.exp(-1j * chi) * o_eth_pol_map,
            spin_n=2 - dn,
            spin_m=-4 - dm,
        )
        pup_field = Field(
            -rho / 4.0 * np.exp(1j * chi) * eth_pol_map, spin_n=2 + dn, spin_m=-4 + dm
        )
        signal_fields = SignalFields(
            spin_00_field,
            spin_p2m4_field,
            spin_m2p4_field,
            t1_field,
            t1_field.conj(),
            pup_field,
            pup_field.conj(),
            pdown_field,
            pdown_field.conj(),
        )
        signal_fields.field_name = "hwp_wedge_field"
        s_00 = signal_fields.get_coupled_field(scan_field, spin_n_out=0, spin_m_out=0)
        sp2m4 = signal_fields.get_coupled_field(scan_field, spin_n_out=2, spin_m_out=-4)
        if mdim == 3:
            fields = np.array([s_00, sp2m4, sp2m4.conj()])
        elif mdim == 5:
            s_t = signal_fields.get_coupled_field(
                scan_field, spin_n_out=dn, spin_m_out=dm
            )
            fields = np.array([s_00, s_t, s_t.conj(), sp2m4, sp2m4.conj()])
        elif mdim == 9:
            s_t = signal_fields.get_coupled_field(
                scan_field, spin_n_out=dn, spin_m_out=dm
            )
            s_down = signal_fields.get_coupled_field(
                scan_field, spin_n_out=2 - dn, spin_m_out=-4 - dm
            )
            s_up = signal_fields.get_coupled_field(
                scan_field, spin_n_out=2 + dn, spin_m_out=-4 + dm
            )
            fields = np.array(
                [
                    s_00,
                    s_t,
                    s_t.conj(),
                    sp2m4,
                    sp2m4.conj(),
                    s_up,
                    s_up.conj(),
                    s_down,
                    s_down.conj(),
                ]
            )
        else:
            raise ValueError("mdim is 3,5 and 9 only supported")
        signal_fields.build_linear_system(fields)
        return signal_fields

    @staticmethod
    def beam_field(
        scan_field,
        mdim: int,
        alm: np.ndarray,
        blm: np.ndarray,
        use_hwp=False,
        spin_k=(0, 2, 4),
        mueller=None,
        lowell_hwp="shifted",
    ):
        """Get the convolved signal field for a general asymmetric beam.

        Args:
            scan_field (:class:`.ScanFields`): scan field instance

            mdim (`int`): dimension of the map-making liner system

            alm (`np.ndarray`): spherical harmonic expansion coefficients of sky

            blm (`np.ndarray`): spherical harmonic expansion coefficients of the
                beam. The beam need not have an elliptical profile.

            use_hwp (`bool`): whether the observation uses HWP or not

            spin_k (`sequence`): non-negative output crossing-angle spins to retain.
                The spin `-k` fields are added as the complex conjugates. The scan field
                must contain the cross-links up to spin ``max(spin_k) + 2`` (``mdim`` = 2, 3)
                or ``max(spin_k) + 4`` (``mdim`` = 5).
                With HWP the input beam order is ``k + spin_m``, so retaining
                all beam orders through ``mmax`` can require output spins
                through ``mmax + 4``. For asymmetric beams, ``range(5)`` retains
                both odd and even moments through spin 4; increase the range
                if additional moments are needed.

            mueller (`np.ndarray` or `None`): real 4x4 Stokes Mueller matrix of
                the rotating HWP. Requires ``use_hwp=True``; defaults to an ideal
                HWP. Uses the shifted-beam narrow-beam approximation. TEBV inputs
                are supported; a missing V component is zero.

            lowell_hwp (`str`): ``"full_sum"`` adds I-input ell=0,1 2f terms
                using the beam multipole sum. Default ``"shifted"`` preserves
                the original approximation. Extra n=0,+/-1 fields needed for
                the correction are included; the scan must contain their
                cross-links, including odd crossing moments for asymmetric beams.
                Other signal terms still follow ``spin_k``. Use ``spin_k=range(5)``
                to also retain their odd moments through spin 4.

        With HWP the polarization basis is (2,-4) and its conjugate; mdim=5
        also solves (4,-4) and its conjugate. Cross-links must cover the
        differences between these output pairs and the active signal pairs.
        A general matrix can require HWP moments +/-2, +/-4, +/-6 and +/-8.

        Returns:
            signal_fields (:class:`.SignalFields`): beam convolution field of the detector
        """
        spin_k = sorted(set(abs(int(k)) for k in spin_k) | {0})
        alm_conv = Convolver(
            alm=alm, nside=scan_field.nside, spin_k=spin_k,
            use_hwp=use_hwp, mueller=mueller, lowell_hwp=lowell_hwp
        )
        blm_conv = Convolver(
            alm=blm, nside=scan_field.nside, spin_k=spin_k, use_hwp=use_hwp
        )
        all_maps = alm_conv * blm_conv
        if use_hwp:
            # Convolver already includes both indices of F_{-n,-h}=conj(F_{n,h}).
            fields = [
                Field(value, spin_n=n, spin_m=h)
                for (n, h), value in all_maps.items() if np.any(value)
            ]
            if not fields:
                fields = [Field(np.zeros(scan_field.npix, dtype=complex), 0, 0)]
        else:
            # d(psi) = sum_k F_k exp(i k psi), F_-k = conj(F_k).
            fields = []
            for k_idx, k in enumerate(spin_k):
                field_k = all_maps[k_idx][0] + all_maps[k_idx][1] / 2
                if k == 0:
                    fields.append(Field(field_k, spin_n=0, spin_m=0))
                else:
                    fields.append(Field(field_k, spin_n=k, spin_m=0))
                    fields.append(Field(field_k.conj(), spin_n=-k, spin_m=0))
        signal_fields = SignalFields(*fields)

        s_0 = signal_fields.get_coupled_field(scan_field, spin_n_out=0, spin_m_out=0)
        pol_harmonic = -4 if use_hwp else 0
        sp2 = signal_fields.get_coupled_field(
            scan_field, spin_n_out=2, spin_m_out=pol_harmonic
        )
        signal_fields.field_name = "beam_field"
        if mdim == 2:
            fields = [sp2, sp2.conj()]
        elif mdim == 3:
            fields = [s_0, sp2, sp2.conj()]
        elif mdim == 5:
            sp4 = signal_fields.get_coupled_field(
                scan_field, spin_n_out=4, spin_m_out=pol_harmonic
            )
            fields = [s_0, sp2, sp2.conj(), sp4, sp4.conj()]
        else:
            raise ValueError("mdim is 2,3 and 5 only supported")
        signal_fields.build_linear_system(fields)
        return signal_fields

    # Compatibility with the original name.
    elliptical_beam_field = beam_field
