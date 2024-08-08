import unittest
import numpy as np
from pathlib import Path
from sbm import ScanFields, SignalFields, Field
import healpy as hp
import os

class TestSBM(unittest.TestCase):
    def setUp(self):
        print(f"Current directory: {os.getcwd()}")
        self.base_path = "maps"
        self.scan_field = ScanFields.load_det(self.base_path, "nside128_boresight")
        self.input_map = hp.read_map("maps/cmb_0000_nside_128_seed_33.fits", field=(0,1,2)) * 1e6
        self.nside = hp.npix2nside(len(self.input_map[0]))

    def test_diff_gain(self, save_output_map=False):
        g_a = 0.01
        g_b = 0.0
        delta_g = g_a - g_b
        I = self.input_map[0]
        P = self.input_map[1] + 1j*self.input_map[2]

        print(f"delta_g: {delta_g}")
        print(f"I.shape: {I.shape}")
        print(f"P.shape: {P.shape}")

        signal_field = SignalFields(
            Field(delta_g*I/2, spin=0),
            Field((2.0+g_a+g_b)*P/4, spin=2),
            Field((2.0+g_a+g_b)*P.conj()/4, spin=-2),
        )
        mdim = 2
        output_map = self.scan_field.map_make(signal_field, mdim=mdim)
        print(f"output_map.shape: {output_map.shape}")

        if save_output_map==True:
            hp.write_map("tests/reference/diff_gain_output_map.fits", output_map)
        else:
            reference = hp.read_map("tests/reference/diff_gain_output_map.fits", field=(0,1,2))
            print(f"reference.shape: {reference.shape}")
            self.assertTrue(np.allclose(output_map, reference))

    def test_diff_pointing(self, save_output_map=False):
        I = self.input_map[0]
        P = self.input_map[1] + 1j*self.input_map[2]
        dI = hp.alm2map_der1(hp.map2alm(self.input_map[0]), nside=self.nside)
        dQ = hp.alm2map_der1(hp.map2alm(self.input_map[1]), nside=self.nside)
        dU = hp.alm2map_der1(hp.map2alm(self.input_map[2]), nside=self.nside)

        eth_I = dI[2] - dI[1]*1j
        eth_P = dQ[2] + dU[1] - (dQ[1] - dU[2])*1j

        rho = np.deg2rad(1/60)
        chi = np.deg2rad(0)
        print("rho: ", rho)
        print("chi: ", chi)

        spin_0_field  = Field(I, spin=0)
        spin_1_field  = Field(-rho/2*np.exp(1j*chi)*eth_I, spin=1)
        spin_m1_field = spin_1_field.conj()
        spin_2_field  = Field(P/2.0, spin=2)
        spin_m2_field = spin_2_field.conj()
        spin_3_field  = Field(-rho/4*np.exp(1j*chi)*eth_P, spin=3)
        spin_m3_field = spin_3_field.conj()

        signal_field = SignalFields(
            spin_0_field,
            spin_1_field,
            spin_m1_field,
            spin_2_field,
            spin_m2_field,
            spin_3_field,
            spin_m3_field,
        )
        mdim = 2
        output_map = self.scan_field.map_make(signal_field, mdim=mdim)
        print(f"output_map.shape: {output_map.shape}")

        if save_output_map==True:
            hp.write_map("tests/reference/diff_pointing_output_map.fits", output_map)
        else:
            reference = hp.read_map("tests/reference/diff_pointing_output_map.fits", field=(0,1,2))
            print(f"reference.shape: {reference.shape}")
            self.assertTrue(np.allclose(output_map, reference))

if __name__ == '__main__':
    unittest.main()
