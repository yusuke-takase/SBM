import unittest
import numpy as np
from pathlib import Path
import sbm
from sbm import ScanFields, SignalFields, Field
import healpy as hp
import os

class TestSBM(unittest.TestCase):
    def setUp(self):
        print(f"Current directory: {os.getcwd()}")
        self.scan_field = ScanFields.load_det("nside_32_boresight_hwp", "tests")
        inputmap = sbm.generate_cmb(128, r=0., cmb_seed=33)
        self.input_map = hp.ud_grade(inputmap, self.scan_field.nside)
        self.nside = hp.npix2nside(len(self.input_map[0]))
        self.I = self.input_map[0]
        self.P = self.input_map[1] + 1j*self.input_map[2]
        dI = hp.alm2map_der1(hp.map2alm(self.input_map[0]), nside=self.nside)
        dQ = hp.alm2map_der1(hp.map2alm(self.input_map[1]), nside=self.nside)
        dU = hp.alm2map_der1(hp.map2alm(self.input_map[2]), nside=self.nside)
        self.eth_I = dI[2] - 1j*dI[1]
        self.eth_P = dQ[2] + dU[1] - 1j*(dQ[1] - dU[2])
        self.o_eth_P = dQ[2] - dU[1] + 1j*(dQ[1] + dU[2])

    def test_diff_gain(self, save_output_map=False):
        g_a = 0.01
        g_b = 0.0
        delta_g = g_a - g_b

        signal_field = self.scan_field.diff_gain_field(g_a, g_b, self.I, self.P)

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
        rho_T = np.deg2rad(1/60)
        chi_T = np.deg2rad(0)
        rho_B = np.deg2rad(0)
        chi_B = np.deg2rad(0)

        signal_field = self.scan_field.diff_pointing_field(rho_T, chi_T, rho_B, chi_B, self.P, self.eth_I, self.eth_P, self.o_eth_P)

        mdim = 2
        output_map = self.scan_field.map_make(signal_field, mdim=mdim)
        print(f"output_map.shape: {output_map.shape}")

        if save_output_map==True:
            hp.write_map("tests/reference/diff_pointing_output_map.fits", output_map, overwrite=True)
            print("Diff pointing output map is saved.")
        else:
            reference = hp.read_map("tests/reference/diff_pointing_output_map.fits", field=(0,1,2))
            print(f"reference.shape: {reference.shape}")
            self.assertTrue(np.allclose(output_map, reference))

    def test_abs_pointing(self, save_output_map=False):
        rho = np.deg2rad(1/60)
        chi = np.deg2rad(0)

        signal_field = self.scan_field.abs_pointing_field(rho, chi, self.I, self.P, self.eth_I, self.eth_P, self.o_eth_P)
        mdim = 3
        output_map = self.scan_field.map_make(signal_field, mdim=mdim)
        print(f"output_map.shape: {output_map.shape}")

        if save_output_map==True:
            hp.write_map("tests/reference/abs_pointing_output_map.fits", output_map, overwrite=True)
            print("Absolute pointing offset with HWP output map is saved.")
        else:
            reference = hp.read_map("tests/reference/abs_pointing_output_map.fits", field=(0,1,2))
            print(f"reference.shape: {reference.shape}")
            self.assertTrue(np.allclose(output_map, reference))

    def test_hwpip(self, save_output_map=False):
        epsilon = 1e-5
        phi_qi = 0.0
        signal_field = self.scan_field.hwp_ip_field(epsilon, phi_qi, self.I)

        mdim = 3
        output_map = self.scan_field.map_make(signal_field, mdim=mdim)
        print(f"output_map.shape: {output_map.shape}")

        if save_output_map==True:
            hp.write_map("tests/reference/hwpip_output_map.fits", output_map, overwrite=True)
            print("HWP-IP output map is saved.")
        else:
            reference = hp.read_map("tests/reference/hwpip_output_map.fits", field=(0,1,2))
            print(f"reference.shape: {reference.shape}")
            self.assertTrue(np.allclose(output_map, reference))

    def test_noise_generation_with_HWP(self, save_output_map=False):
        net_ukrts = 100
        self.scan_field.generate_noise_pdf(net_ukrts=net_ukrts)
        mdim = 3
        seed = 12345
        use_hwp = True
        output_map = self.scan_field.generate_noise(mdim, use_hwp, seed)

        if save_output_map==True:
            hp.write_map(f"tests/reference/noise_map_{net_ukrts}ukrts_hwp.fits", output_map, overwrite=True)
            print("Noise map is saved.")
        else:
            reference = hp.read_map(f"tests/reference/noise_map_{net_ukrts}ukrts_hwp.fits", field=(0,1,2))
            print(f"reference.shape: {reference.shape}")
            self.assertTrue(np.allclose(output_map, reference))

    def test_noise_generation_without_HWP(self, save_output_map=False):
        net_ukrts = 100
        self.scan_field.generate_noise_pdf(net_ukrts=net_ukrts)
        mdim = 3
        seed = 12345
        use_hwp = False
        output_map = self.scan_field.generate_noise(mdim, use_hwp, seed)

        if save_output_map==True:
            hp.write_map(f"tests/reference/noise_map_{net_ukrts}ukrts.fits", output_map, overwrite=True)
            print("Noise map is saved.")
        else:
            reference = hp.read_map(f"tests/reference/noise_map_{net_ukrts}ukrts.fits", field=(0,1,2))
            print(f"reference.shape: {reference.shape}")
            self.assertTrue(np.allclose(output_map, reference))

if __name__ == '__main__':
    unittest.main()
