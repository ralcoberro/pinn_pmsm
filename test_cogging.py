"""
Unit tests for the analytical cogging torque functions in femm_utils.py.

Run with:  python test_cogging.py
"""
import sys
import os
import csv
import math
import unittest

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Stub the femm module so we can import femm_utils without pyfemm installed
sys.modules['femm'] = type(sys)('femm')

import numpy as np
from femm_utils import (
    cogging_params,
    slot_opening_width,
    carter_coefficient,
    flux_density_harmonics,
    permeance_fourier,
    cogging_torque_zhu,
    cogging_torque_from_row,
)


# Fixed machine parameters (18-slot / 24-pole outrunner, N52 magnets)
Ns = 18
POLES = 24
Br = 1.45
MU_R = 1.05
KL = 0.98


class TestCoggingParams(unittest.TestCase):

    def test_18s_24p_defaults(self):
        Nc, alpha_p, kso = cogging_params(Ns, POLES)
        self.assertEqual(Nc, 72)
        self.assertAlmostEqual(alpha_p, 1.0)
        self.assertAlmostEqual(kso, 0.25)

    def test_lcm_different_topologies(self):
        Nc_12s12p, _, _ = cogging_params(12, 12)
        self.assertEqual(Nc_12s12p, 12)

        Nc_12s10p, _, _ = cogging_params(12, 10)
        self.assertEqual(Nc_12s10p, 60)

    def test_alpha_p_clamped_below_one(self):
        _, alpha_p, _ = cogging_params(Ns, POLES, k1=0.0, k2=0.5)
        self.assertLessEqual(alpha_p, 1.0)


class TestSlotOpening(unittest.TestCase):

    def test_positive_and_proportional(self):
        _, _, kso = cogging_params(Ns, POLES)
        b1 = slot_opening_width(0.050, 0.0008, Ns, kso)
        b2 = slot_opening_width(0.060, 0.0008, Ns, kso)
        self.assertGreater(b1, 0)
        self.assertGreater(b2, b1)

    def test_matches_pmsm_formula(self):
        """b_so = 2*pi*(R_g - g/2) / Ns * kso"""
        R_g, g = 0.0507, 0.0008
        _, _, kso = cogging_params(Ns, POLES)
        expected = 2 * math.pi * (R_g - g / 2) / Ns * kso
        self.assertAlmostEqual(slot_opening_width(R_g, g, Ns, kso), expected)


class TestCarterCoefficient(unittest.TestCase):

    def test_greater_than_one(self):
        kc = carter_coefficient(0.004, 0.018, 0.002)
        self.assertGreater(kc, 1.0)

    def test_approaches_one_for_small_opening(self):
        kc = carter_coefficient(1e-6, 0.018, 0.002)
        self.assertAlmostEqual(kc, 1.0, places=4)

    def test_increases_with_slot_opening(self):
        kc_small = carter_coefficient(0.002, 0.018, 0.002)
        kc_large = carter_coefficient(0.008, 0.018, 0.002)
        self.assertGreater(kc_large, kc_small)


class TestFluxDensityHarmonics(unittest.TestCase):

    def test_fundamental_positive(self):
        B = flux_density_harmonics(Br, MU_R, 0.0015, 0.0008, 1.0, KL)
        self.assertGreater(B[0], 0)

    def test_decreasing_amplitude(self):
        B = flux_density_harmonics(Br, MU_R, 0.0015, 0.0008, 1.0, KL)
        self.assertGreater(abs(B[0]), abs(B[1]))
        self.assertGreater(abs(B[1]), abs(B[2]))

    def test_alpha_p_one_all_odd_nonzero(self):
        """With full pole coverage, all odd harmonics exist."""
        B = flux_density_harmonics(Br, MU_R, 0.0015, 0.0008, 1.0, KL, n_max=5)
        for i in range(5):
            self.assertNotAlmostEqual(B[i], 0.0, places=4)

    def test_output_shape(self):
        B = flux_density_harmonics(Br, MU_R, 0.0015, 0.0008, 1.0, n_max=10)
        self.assertEqual(B.shape, (10,))


class TestPermeanceFourier(unittest.TestCase):

    def test_g0_equals_inverse_carter(self):
        b_so, tau_s, g_prime = 0.004, 0.018, 0.002
        G = permeance_fourier(b_so, tau_s, g_prime)
        kc = carter_coefficient(b_so, tau_s, g_prime)
        self.assertAlmostEqual(G[0], 1.0 / kc, places=10)

    def test_higher_harmonics_decay(self):
        G = permeance_fourier(0.004, 0.018, 0.002, n_max=20)
        self.assertGreater(abs(G[1]), abs(G[10]))

    def test_output_shape(self):
        G = permeance_fourier(0.004, 0.018, 0.002, n_max=15)
        self.assertEqual(G.shape, (16,))


class TestCoggingTorqueZhu(unittest.TestCase):

    def _sample_params(self):
        R_g, g, h_m, L_stk = 0.0507, 0.0008, 0.00146, 0.02
        _, alpha_p, kso = cogging_params(Ns, POLES)
        b_so = slot_opening_width(R_g, g, Ns, kso)
        return dict(Ns=Ns, poles=POLES, Br=Br, mu_r=MU_R,
                    h_m=h_m, g=g, alpha_p=alpha_p, b_so=b_so,
                    R_g=R_g, L_stk=L_stk, kl=KL)

    def test_positive_result(self):
        T = cogging_torque_zhu(**self._sample_params())
        self.assertGreater(T, 0)

    def test_order_of_magnitude(self):
        """No-load cogging should be well below 1 N·m for these sizes."""
        T = cogging_torque_zhu(**self._sample_params())
        self.assertLess(T, 1.0)
        self.assertGreater(T, 0.001)

    def test_scales_with_stack_length(self):
        params = self._sample_params()
        T1 = cogging_torque_zhu(**params)
        params['L_stk'] *= 2
        T2 = cogging_torque_zhu(**params)
        self.assertAlmostEqual(T2 / T1, 2.0, places=1)


class TestCoggingTorqueFromRow(unittest.TestCase):

    DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
    DESIGNS_CSV = os.path.join(DATASET_DIR, 'designs.csv')
    TORQUE_CSV = os.path.join(DATASET_DIR, 'torque_results.csv')

    @unittest.skipUnless(
        os.path.isfile(os.path.join(os.path.dirname(__file__),
                                    'dataset', 'designs.csv')),
        'dataset/designs.csv not found')
    def test_first_row(self):
        with open(self.DESIGNS_CSV) as f:
            row = next(csv.DictReader(f))
        T = cogging_torque_from_row(row)
        self.assertFalse(math.isnan(T))
        self.assertGreater(T, 0)

    def test_nan_on_bad_data(self):
        bad_row = {'slots': '18', 'pp': '12',
                   'magnet_thickness': 'nan', 'airgap_thickness': '0.0008',
                   'airgap_radius': '0.05', 'stator_length': '0.02'}
        T = cogging_torque_from_row(bad_row)
        self.assertTrue(math.isnan(T))

    @unittest.skipUnless(
        os.path.isfile(os.path.join(os.path.dirname(__file__),
                                    'dataset', 'designs.csv')),
        'dataset/designs.csv not found')
    def test_batch_no_crash(self):
        """All rows should return a finite float or NaN, never raise."""
        with open(self.DESIGNS_CSV) as f:
            for row in csv.DictReader(f):
                T = cogging_torque_from_row(row)
                self.assertIsInstance(T, float)

    @unittest.skipUnless(
        os.path.isfile(os.path.join(os.path.dirname(__file__),
                                    'dataset', 'torque_results.csv')),
        'dataset/torque_results.csv not found')
    def test_correlation_report(self):
        """Not a pass/fail assertion — prints the correlation summary."""
        zhu, fem = [], []
        with open(self.DESIGNS_CSV) as fd, \
             open(self.TORQUE_CSV) as ft:
            for d, t in zip(csv.DictReader(fd), csv.DictReader(ft)):
                tc = cogging_torque_from_row(d)
                h6 = float(t['h6_amp'])
                if not math.isnan(tc):
                    zhu.append(tc)
                    fem.append(h6)

        zhu = np.array(zhu)
        fem = np.array(fem)
        pearson = np.corrcoef(zhu, fem)[0, 1]

        print(f"\n  Valid samples: {len(zhu)}")
        print(f"  Zhu-Howe  — mean: {zhu.mean():.4f}, "
              f"std: {zhu.std():.4f}")
        print(f"  FEMM H6   — mean: {fem.mean():.4f}, "
              f"std: {fem.std():.4f}")
        print(f"  Mean ratio (Zhu/FEMM): {(zhu / fem).mean():.3f}")
        print(f"  Pearson correlation:    {pearson:.4f}")
        self.assertEqual(len(zhu), len(fem))


if __name__ == '__main__':
    unittest.main(verbosity=2)
