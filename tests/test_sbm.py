import unittest
import numpy as np
import sbm
from sbm import ScanFields, SignalFields
import healpy as hp
import os
from pathlib import Path

save = False


class TestSBM(unittest.TestCase):
    def setUp(self):
        print(f"Current directory: {os.getcwd()}")
        self.scan_field = ScanFields.load_det("nside_32_boresight_hwp", "tests")
        inputmap = sbm.generate_cmb(128, r=0.0, cmb_seed=33)
        self.input_map = hp.ud_grade(inputmap, self.scan_field.nside)
        self.nside = hp.npix2nside(len(self.input_map[0]))
        self.I = self.input_map[0]
        self.P = self.input_map[1] + 1j * self.input_map[2]
        dI = hp.alm2map_der1(hp.map2alm(self.input_map[0]), nside=self.nside)
        dQ = hp.alm2map_der1(hp.map2alm(self.input_map[1]), nside=self.nside)
        dU = hp.alm2map_der1(hp.map2alm(self.input_map[2]), nside=self.nside)
        self.eth_I = dI[2] - 1j * dI[1]
        self.eth_P = dQ[2] + dU[1] - 1j * (dQ[1] - dU[2])
        self.o_eth_P = dQ[2] - dU[1] + 1j * (dQ[1] + dU[2])

    def test_diff_gain(self, save_output_map=save):
        g_a = 0.01
        g_b = 0.0
        mdims = [2, 3]
        for mdim in mdims:
            signal_field = SignalFields.diff_gain_field(
                self.scan_field, mdim, g_a, g_b, self.I, self.P
            )
            output_map = self.scan_field.map_make(signal_field)
            if save_output_map is True:
                hp.write_map(
                    f"tests/reference/diff_gain_output_map_mdim_{mdim}.fits",
                    output_map,
                    overwrite=True,
                )
                print("Diff gain output map is saved.")
            else:
                reference = hp.read_map(
                    f"tests/reference/diff_gain_output_map_mdim_{mdim}.fits",
                    field=(0, 1, 2),
                )
                self.assertTrue(np.allclose(output_map, reference))

    def test_elliptical_beam(self, save_output_map=save):
        alm = hp.map2alm(self.input_map)
        q = 0.9
        # the beam must be resolved at nside=32 (pixel size ~1.8 deg)
        fwhm = np.deg2rad(5.0)
        beam = sbm.elliptical_beam(self.nside, fwhm, q)
        blm = hp.map2alm(beam)
        mdims = [2, 3]
        for mdim in mdims:
            # The original name remains supported for existing callers.
            signal_field = SignalFields.elliptical_beam_field(
                self.scan_field, mdim, alm, blm
            )
            output_map = self.scan_field.map_make(signal_field)
            if save_output_map is True:
                hp.write_map(
                    f"tests/reference/elliptical_beam_conv_output_map_mdim_{mdim}.fits",
                    output_map,
                    overwrite=True,
                )
                print("Beam convolved output map is saved.")
            else:
                reference = hp.read_map(
                    f"tests/reference/elliptical_beam_conv_output_map_mdim_{mdim}.fits",
                    field=(0, 1, 2),
                )
                self.assertTrue(np.allclose(output_map, reference))

    def test_diff_pointing(self, save_output_map=save):
        rho_T = np.deg2rad(1 / 60)
        chi_T = np.deg2rad(0)
        rho_B = np.deg2rad(0)
        chi_B = np.deg2rad(0)
        mdims = [2, 4, 6]
        for mdim in mdims:
            signal_field = SignalFields.diff_pointing_field(
                self.scan_field,
                mdim,
                rho_T,
                chi_T,
                rho_B,
                chi_B,
                self.P,
                self.eth_I,
                self.eth_P,
                self.o_eth_P,
            )
            output_map = self.scan_field.map_make(signal_field)
            if save_output_map is True:
                hp.write_map(
                    f"tests/reference/diff_pointing_output_map_mdim_{mdim}.fits",
                    output_map,
                    overwrite=True,
                )
                print("Diff pointing output map is saved.")
            else:
                reference = hp.read_map(
                    f"tests/reference/diff_pointing_output_map_mdim_{mdim}.fits",
                    field=(0, 1, 2),
                )
                self.assertTrue(np.allclose(output_map, reference))

    def test_abs_pointing(self, save_output_map=save):
        rho = np.deg2rad(1 / 60)
        chi = np.deg2rad(0)
        mdims = [3, 5, 9]
        for mdim in mdims:
            signal_field = SignalFields.abs_pointing_field(
                self.scan_field,
                mdim,
                rho,
                chi,
                self.I,
                self.P,
                self.eth_I,
                self.eth_P,
                self.o_eth_P,
            )
            output_map = self.scan_field.map_make(signal_field)

            if save_output_map is True:
                hp.write_map(
                    f"tests/reference/abs_pointing_output_map_mdim_{mdim}.fits",
                    output_map,
                    overwrite=True,
                )
                print("Absolute pointing offset with HWP output map is saved.")
            else:
                reference = hp.read_map(
                    f"tests/reference/abs_pointing_output_map_mdim_{mdim}.fits",
                    field=(0, 1, 2),
                )
                self.assertTrue(np.allclose(output_map, reference))

    def test_random_pointing(self):
        # The standard scan fixture stops at |n|=6 and |m|=8. Random pointing
        # also needs the sums of its moment spins, up to |n|=10 and |m|=16.
        reference_path = Path(__file__).parent / "reference" / "random_pointing.npz"
        with np.load(reference_path, allow_pickle=False) as reference:
            scan_field = ScanFields()
            scan_field.hitmap = reference["hitmap"].copy()
            scan_field.npix = len(scan_field.hitmap)
            scan_field.nside = hp.npix2nside(scan_field.npix)
            scan_field.spins_n = np.arange(11)
            scan_field.spins_m = np.arange(-16, 17)
            scan_field.h = np.zeros((scan_field.npix, 33, 11), dtype=complex)
            for (n, m), values in zip(
                reference["spins"], reference["harmonics"]
            ):
                scan_field.h[:, m + 16, n] = values
                if n == 0:
                    scan_field.h[:, -m + 16, n] = values.conj()

            signal_field = SignalFields.random_pointing_field(
                scan_field,
                mdim=3,
                sigma_rad=float(reference["sigma_rad"]),
                pol_map=reference["P"],
                eth_pol_map=reference["eth_P"],
                o_eth_pol_map=reference["o_eth_P"],
                seed=int(reference["seed"]),
            )
            self.assertIsInstance(signal_field, SignalFields)
            self.assertEqual(signal_field.field_name, "random_pointing_field")
            np.testing.assert_array_equal(signal_field.spin_n_basis, [0, -2, 2])
            np.testing.assert_array_equal(signal_field.spin_m_basis, [0, 4, -4])
            np.testing.assert_allclose(
                scan_field.map_make(signal_field),
                reference["output_iqu"],
                rtol=1e-12,
                atol=1e-12,
            )
            reconstructed = scan_field.map_make(signal_field, only_iqu=False)
            np.testing.assert_allclose(
                reconstructed.extract_iqu(), reference["output_iqu"], atol=1e-12
            )

    def test_hwpip(self, save_output_map=save):
        epsilon = 1e-5
        phi_qi = 0.0
        mdims = [3]
        for mdim in mdims:
            signal_field = SignalFields.hwp_ip_field(
                self.scan_field, mdim, epsilon, phi_qi, self.I
            )
            output_map = self.scan_field.map_make(signal_field)
            if save_output_map is True:
                hp.write_map(
                    f"tests/reference/hwpip_output_map_mdim_{mdim}.fits",
                    output_map,
                    overwrite=True,
                )
                print("HWP-IP output map is saved.")
            else:
                reference = hp.read_map(
                    f"tests/reference/hwpip_output_map_mdim_{mdim}.fits",
                    field=(0, 1, 2),
                )
                self.assertTrue(np.allclose(output_map, reference))

    def test_noise_generation_with_HWP(self, save_output_map=save):
        spin_n_basis_list = [
            [2, -2],
            [0, 2, -2],
            [1, -1, 2, -2],
        ]
        spin_m_basis_list = [
            [0, 0],
            [0, 0, 0],
            [0, 0, 0, 0],
        ]
        seed = 12345
        net_ukrts = 100
        self.scan_field.generate_noise_pdf(net_ukrts=net_ukrts)
        for spin_n_basis, spin_m_basis in zip(spin_n_basis_list, spin_m_basis_list):
            mdim = len(spin_n_basis)
            output_map = self.scan_field.generate_noise(
                spin_n_basis, spin_m_basis, seed
            )
            if save_output_map is True:
                hp.write_map(
                    f"tests/reference/noise_map_{net_ukrts}ukrts_hwp_mdim_{mdim}.fits",
                    output_map,
                    overwrite=True,
                )
                print("Noise map is saved.")
            else:
                reference = hp.read_map(
                    f"tests/reference/noise_map_{net_ukrts}ukrts_hwp_mdim_{mdim}.fits",
                    field=(0, 1, 2),
                )
                self.assertTrue(np.allclose(output_map, reference))

    def test_noise_generation_without_HWP(self, save_output_map=save):
        spin_n_basis_list = [
            [0, 2, -2],
            [0, 1, -1, 2, -2],
        ]
        spin_m_basis_list = [
            [0, -4, 4],
            [0, 0, 0, -4, 4],
        ]
        net_ukrts = 100
        seed = 12345
        self.scan_field.generate_noise_pdf(net_ukrts=net_ukrts)
        for spin_n_basis, spin_m_basis in zip(spin_n_basis_list, spin_m_basis_list):
            mdim = len(spin_n_basis)
            output_map = self.scan_field.generate_noise(
                spin_n_basis, spin_m_basis, seed
            )
            if save_output_map is True:
                hp.write_map(
                    f"tests/reference/noise_map_{net_ukrts}ukrts_mdim_{mdim}.fits",
                    output_map,
                    overwrite=True,
                )
                print("Noise map is saved.")
            else:
                reference = hp.read_map(
                    f"tests/reference/noise_map_{net_ukrts}ukrts_mdim_{mdim}.fits",
                    field=(0, 1, 2),
                )
                self.assertTrue(np.allclose(output_map, reference))


if __name__ == "__main__":
    unittest.main()
