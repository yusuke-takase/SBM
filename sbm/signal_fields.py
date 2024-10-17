import h5py
import numpy as np
import healpy as hp
import os
import copy
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd
import litebird_sim as lbs
from litebird_sim import Imo
from pathlib import Path
import toml

class Field:
    """ Class to store the field data of detectors """
    def __init__(self, field: np.ndarray, spin_n: float, spin_m: float):
        """ Initialize the class with field and spin data

        Args:
            field (np.ndarray): field data of the detector

            spin_n (float): spin number of the crossing angle

            spin_m (float): spin number of the HWP angle
        """
        if all(isinstance(x, float) for x in field):
            self.field = field + 1j*np.zeros(len(field))
        else:
            self.field = field
        self.spin_n = spin_n
        self.spin_m = spin_m

    def conj(self):
        """ Get the complex conjugate of the field """
        return Field(self.field.conj(), -self.spin_n, -self.spin_m)

    def __add__(self, other):
        """ Add the field of two detectors

        Args:
            other (Field): field of the other detector

        Returns:
            result (Field): sum of the fields of the two detectors
        """
        if not isinstance(other, Field):
            return NotImplemented
        result = copy.deepcopy(self)
        result.field += other.field
        return result

class SignalFields:
    """ Class to store the signal fields data of detectors """
    def __init__(self, *fields: Field):
        """ Initialize the class with field data

        Args:
            fields (Field): field (map) data of the signal
        """
        self.fields = sorted(fields, key=lambda field: (field.spin_n, field.spin_m))
        self.spins_n = np.array([field.spin_n for field in self.fields])
        self.spins_m = np.array([field.spin_m for field in self.fields])
        self.field_name = None
        self.coupled_fields = None
        self.spin_n_basis = []
        self.spin_m_basis = []

    def get_field(self, spin_n: float, spin_m: float):
        """ Get the field of the given spin number

        Args:
            spin_n (float): spin number for which the field is to be obtained

            spin_m (float): spin number for which the field is to be obtained

        Returns:
            field (np.ndarray): field of the detector for the given spin number
        """
        for i, field in enumerate(self.fields):
            if field.spin_n == spin_n and field.spin_m == spin_m:
                return field.field
        return None

    def __add__(self, other):
        """ Add the signal fieldd

        Args:
            other (SignalFields): signal fields

        Returns:
            result (SignalFields): sum of the signal fields
        """
        if not isinstance(other, SignalFields):
            return NotImplemented
        result = copy.deepcopy(self)
        for i in range(len(self.spins_n)): # この足し方でいいのか？
            result.fields[i].field += other.fields[i].field
        return result

    def get_coupled_field(self, scan_field, spin_n_out: float, spin_m_out: float):
        """ Multiply the scan fields and signal fields to get the detected fields by
        given cross-linking

        Args:
            scan_fields (ScanFields): scan fields instance

            spin_n_out (float): spin_n of the output field

            spin_m_out (float): spin_m of the output field

        Returns:
            results (np.ndarray): detected fields by the given cross-linking
        """
        results = []
        for i in range(len(self.spins_n)):
            n = self.spins_n[i]
            m = self.spins_m[i]
            delta_n = spin_n_out - n
            delta_m = spin_m_out - m
            hS = scan_field.get_xlink(delta_n,delta_m) * self.get_field(n,m)
            results.append(hS)
        return np.array(results).sum(0)

    @staticmethod
    def diff_gain_field(
        scan_field,
        mdim: int,
        gain_T: float,
        gain_B: float,
        I: np.ndarray,
        P: np.ndarray,
        ):
        """" Get the differential gain field of the detector

        Args:
            scan_field (ScanFields): scan field instance

            mdim (int): dimension of the map-making liner system

            gain_T (float): gain of the `Top` detector

            gain_B (float): gain of the `Bottom` detector

            I (np.ndarray): temperature map

            P (np.ndarray): polarization map (i.e. Q+iU)

        Returns:
            signal_fields (SignalFields): differential gain field of the detector
        """
        delta_g = gain_T - gain_B
        signal_fields = SignalFields(
            Field(delta_g*I/2.0, spin_n=0, spin_m=0),
            Field((2.0+gain_T+gain_B)*P/4.0, spin_n=2, spin_m=0),
            Field((2.0+gain_T+gain_B)*P.conj()/4.0, spin_n=-2, spin_m=0),
        )
        signal_fields.syst_field_name = "diff_gain_field"
        s_0 = signal_fields.get_coupled_field(scan_field, spin_n_out=0, spin_m_out=0)
        sp2 = signal_fields.get_coupled_field(scan_field, spin_n_out=2, spin_m_out=0)
        if mdim == 2:
            coupled_fields = np.array([sp2/2.0, sp2.conj()/2.0])
            spin_n_basis = [2, -2]
            spin_m_basis = [0 for _ in range(len(spin_n_basis))]
        elif mdim == 3:
            coupled_fields = np.array([s_0, sp2/2.0, sp2.conj()/2.0])
            spin_n_basis = [0, 2, -2]
            spin_m_basis = [0 for _ in range(len(spin_n_basis))]
        else:
            raise ValueError("mdim is 2 and 3 only supported")
        signal_fields.coupled_fields = coupled_fields
        signal_fields.spin_n_basis = spin_n_basis
        signal_fields.spin_m_basis = spin_m_basis
        return signal_fields

    @staticmethod
    def diff_pointing_field(
        scan_field,
        mdim: int,
        rho_T: float,
        rho_B: float,
        chi_T: float,
        chi_B: float,
        P: np.ndarray,
        eth_I: np.ndarray,
        eth_P: np.ndarray,
        o_eth_P: np.ndarray,
        ):
        """ Get the differential pointing field of the detector

        Args:
            scan_field (ScanFields): scan field instance

            mdim (int): dimension of the map-making liner system

            rho_T (float): magnitude of pointing offset of the `Top` detector in radian

            rho_B (float): magnitude of pointing offset of the `Bottom` detector in radian

            chi_T (float): direction of the pointing offset of the `Top` detector in radian

            chi_B (float): direction of the pointing offset of the `Bottom` detector in radian

            P (np.ndarray): polarization map (i.e. Q+iU)

            eth_I (np.ndarray): spin up gradient of the temperature map

            eth_P (np.ndarray): spin up gradient of the polarization map

            o_eth_P (np.ndarray): spin down gradient of the polarization map
        """
        zeta   = rho_T * np.exp(1j*chi_T) - 1j*rho_B * np.exp(1j*chi_B)
        o_zeta = rho_T * np.exp(1j*chi_T) + 1j*rho_B * np.exp(1j*chi_B) #\overline{\zeta}

        spin_1_field  = Field(-1.0/4.0 * (zeta*eth_I + o_zeta.conj()*o_eth_P), spin_n=1, spin_m=0)
        spin_m1_field = spin_1_field.conj()
        spin_2_field  = Field(P/2.0, spin_n=2, spin_m=0)
        spin_m2_field = spin_2_field.conj()
        spin_3_field  = Field(-1.0/4.0 * o_zeta * eth_P, spin_n=3, spin_m=0)
        spin_m3_field = spin_3_field.conj()

        signal_fields = SignalFields(
            spin_1_field,
            spin_m1_field,
            spin_2_field,
            spin_m2_field,
            spin_3_field,
            spin_m3_field,
        )
        signal_fields.syst_field_name = "diff_pointing_field"
        sp2 = signal_fields.get_coupled_field(scan_field, spin_n_out=2, spin_m_out=0)
        if mdim == 2:
            coupled_fields = np.array([sp2/2.0, sp2.conj()/2.0])
            spin_n_basis = [2, -2]
            spin_m_basis = [0 for _ in range(len(spin_n_basis))]
        elif mdim == 4: # T grad. mitigation
            sp1 = signal_fields.get_coupled_field(scan_field, spin_n_out=1, spin_m_out=0)
            coupled_fields = np.array([sp1/2.0, sp1.conj()/2.0, sp2/2.0, sp2.conj()/2.0])
            spin_n_basis = [1, -1, 2, -2]
            spin_m_basis = [0 for _ in range(len(spin_n_basis))]
        elif mdim == 6: # T and P grad. mitigation
            sp1 = signal_fields.get_coupled_field(scan_field, spin_n_out=1, spin_m_out=0)
            sp3 = signal_fields.get_coupled_field(scan_field, spin_n_out=3, spin_m_out=0)
            coupled_fields = np.array([sp1/2.0, sp1.conj()/2.0, sp2/2.0, sp2.conj()/2.0, sp3/2.0, sp3.conj()/2.0])
            spin_n_basis = [1, -1, 2, -2, 3, -3]
            spin_m_basis = [0 for _ in range(len(spin_n_basis))]
        else:
            raise ValueError("mdim is 2,4 and 6 only supported")
        signal_fields.coupled_fields = coupled_fields
        signal_fields.spin_n_basis = spin_n_basis
        signal_fields.spin_m_basis = spin_m_basis
        return signal_fields

    @staticmethod
    def hwp_ip_field(
        scan_field,
        mdim: int,
        epsilon: float,
        phi_qi: float,
        I: np.ndarray,
        ):
        """ Get the HWP instrumental polarization field of the detector

        Args:
            scan_field (ScanFields): scan field instance

            mdim (int): dimension of the map-making liner system

            epsilon (float): amplitude of the HWP Meuller matrix element varying at 4f

            phi_qi (float): phase shift in the HWP

            I (np.ndarray): temperature map
        """
        signal_fields = SignalFields(
            Field(epsilon/2.0 * np.exp(-1j*phi_qi)*I, spin_n=4, spin_m=-4),
            Field(epsilon/2.0 * np.exp(1j*phi_qi)*I, spin_n=-4, spin_m=4),
        )
        signal_fields.syst_field_name = "hwp_ip_field"
        s_00 = signal_fields.get_coupled_field(scan_field, spin_n_out=0, spin_m_out=0)
        sp2m4 = signal_fields.get_coupled_field(scan_field, spin_n_out=2, spin_m_out=-4)
        if mdim == 3:
            coupled_fields = np.array([s_00, sp2m4/2.0, sp2m4.conj()/2.0])
            spin_n_basis = [0, -2, 2]
            spin_m_basis = [0, 4, -4]
        else:
            raise ValueError("mdim is 3 only supported")
        signal_fields.coupled_fields = coupled_fields
        signal_fields.spin_n_basis = spin_n_basis
        signal_fields.spin_m_basis = spin_m_basis
        return signal_fields

    @staticmethod
    def abs_pointing_field(
        scan_field,
        mdim: int,
        rho: float,
        chi: float,
        I: np.ndarray,
        P: np.ndarray,
        eth_I: np.ndarray,
        eth_P: np.ndarray,
        o_eth_P: np.ndarray,
        ):
        """ Get the absolute pointing field of the detector

        Args:
            scan_field (ScanFields): scan field instance

            mdim (int): dimension of the map-making liner system

            rho (float): magnitude of pointing offset in radian

            chi (float): direction of the pointing offset in radian

            I (np.ndarray): temperature map

            P (np.ndarray): polarization map (i.e. Q+iU)

            eth_I (np.ndarray): spin up gradient of the temperature map

            eth_P (np.ndarray): spin up gradient of the polarization map

            o_eth_P (np.ndarray): spin down gradient of the polarization map
        """
        spin_00_field   = Field(I, spin_n=0, spin_m=0)
        spin_p2m4_field = Field(P/2.0, spin_n=2, spin_m=-4)
        spin_m2p4_field = spin_p2m4_field.conj()
        spin_p10_field  = Field(-rho/2.0*np.exp(1j*chi)*eth_I, spin_n=1, spin_m=0)
        spin_m10_field  = spin_p10_field.conj()
        spin_p1m4_field = Field(-rho/4.0*np.exp(-1j*chi)*o_eth_P, spin_n=1, spin_m=-4)
        spin_m1p4_field = spin_p1m4_field.conj()
        spin_p3m4_field = Field(-rho/4.0*np.exp(1j*chi)*eth_P, spin_n=3, spin_m=-4)
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
        signal_fields.syst_field_name = "abs_pointing_field"
        s_00 = signal_fields.get_coupled_field(scan_field, spin_n_out=0, spin_m_out=0)
        sp2m4 = signal_fields.get_coupled_field(scan_field, spin_n_out=2, spin_m_out=-4)
        if mdim == 3:
            coupled_fields = np.array([s_00, sp2m4/2.0, sp2m4.conj()/2.0])
            spin_n_basis = [0, -2, 2]
            spin_m_basis = [0, 4, -4]
        elif mdim == 5:
            sp10 = signal_fields.get_coupled_field(scan_field, spin_n_out=1, spin_m_out=0)
            coupled_fields = np.array([s_00, sp10/2.0, sp10.conj()/2.0, sp2m4/2.0, sp2m4.conj()/2.0])
            spin_n_basis = [0,-1,1,-2, 2]
            spin_m_basis = [0, 0,0, 4,-4]
        elif mdim == 9:
            sp10 = signal_fields.get_coupled_field(scan_field, spin_n_out=1, spin_m_out=0)
            sp3m4 = signal_fields.get_coupled_field(scan_field, spin_n_out=3, spin_m_out=-4)
            sp1m4 = signal_fields.get_coupled_field(scan_field, spin_n_out=1, spin_m_out=-4)
            coupled_fields = np.array([
                s_00,
                sp10/2.0,
                sp10.conj()/2.0,
                sp2m4/2.0,
                sp2m4.conj()/2.0,
                sp3m4/2.0,
                sp3m4.conj()/2.0,
                sp1m4/2.0,
                sp1m4.conj()/2.0,
                ])
            spin_n_basis = [0,-1, 1,-2, 2,-3, 3,-1, 1]
            spin_m_basis = [0, 0, 0, 4,-4, 4,-4, 4,-4]
        else:
            raise ValueError("mdim is 3,5 and 9 only supported")
        signal_fields.coupled_fields = coupled_fields
        signal_fields.spin_n_basis = spin_n_basis
        signal_fields.spin_m_basis = spin_m_basis
        return signal_fields
