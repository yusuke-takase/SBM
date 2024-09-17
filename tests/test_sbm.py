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
        self.scan_field = ScanFields.load_det("nside128_boresight", self.base_path)
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
            hp.write_map("tests/reference/diff_gain_output_map.fits", output_map, overwrite=True)
            print("Diff gain output map is saved.")
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

        eth_I = dI[2] - 1j*dI[1]
        eth_P = dQ[2] + dU[1] - 1j*(dQ[1] - dU[2])
        o_eth_P = dQ[2] - dU[1] + 1j*(dQ[1] + dU[2])

        rho_T = np.deg2rad(1/60)
        chi_T = np.deg2rad(0)
        rho_B = np.deg2rad(0)
        chi_B = np.deg2rad(0)
        zeta   = rho_T * np.exp(1j*chi_T) - 1j*rho_B * np.exp(1j*chi_B)
        o_zeta = rho_T * np.exp(1j*chi_T) + 1j*rho_B * np.exp(1j*chi_B) #\overline{\zeta}

        spin_0_field  = Field(np.zeros(len(P)), spin=0)
        spin_1_field  = Field(-1.0/4.0 * (zeta*eth_I + o_zeta.conj()*o_eth_P), spin=1)
        spin_m1_field = spin_1_field.conj()
        spin_2_field  = Field(P/2.0, spin=2)
        spin_m2_field = spin_2_field.conj()
        spin_3_field  = Field(-1.0/4.0 * o_zeta * eth_P, spin=3)
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
            hp.write_map("tests/reference/diff_pointing_output_map.fits", output_map, overwrite=True)
            print("Diff gain output map is saved.")
        else:
            reference = hp.read_map("tests/reference/diff_pointing_output_map.fits", field=(0,1,2))
            print(f"reference.shape: {reference.shape}")
            self.assertTrue(np.allclose(output_map, reference))

    def test_noise_generation(self, save_output_map=False):
        net_ukrts = 100
        self.scan_field.generate_noise_pdf(net_ukrts=net_ukrts)
        mdim = 3
        seed = 12345
        output_map = self.scan_field.generate_noise(mdim, seed)
        if save_output_map==True:
            hp.write_map(f"tests/reference/noise_map_{net_ukrts}ukrts.fits", output_map, overwrite=True)
            print("Noise map is saved.")
        else:
            reference = hp.read_map(f"tests/reference/noise_map_{net_ukrts}ukrts.fits", field=(0,1,2))
            print(f"reference.shape: {reference.shape}")
            self.assertTrue(np.allclose(output_map, reference))

if __name__ == '__main__':
    unittest.main()
