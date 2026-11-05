"""The spin weights of the sky maps and their check in the signal models.

A component ``S_{n,m}`` has spin weight ``n``: the spin weights of the factors
on its right-hand side add up to ``n`` (``sbm.spin``). The test at the end
checks that every systematic builds its components from maps that carry their
spin weight, so that a wrong couple in a new systematic fails at once.
"""

import pickle
import sys

import numpy as np
import pytest

from sbm import systematics as sy
from sbm.signal_model import SignalModel
from sbm.sky import SECOND_ORDER_KEYS, SECOND_ORDER_SPINS, Sky
from sbm.spin import SpinMap, SpinWeightError, spin_weight
from sbm.testing import RealSky

from .test_systematics import cases

NPIX = 48


@pytest.fixture
def sky():
    return RealSky(2, seed=1).sky()


def test_the_maps_of_a_sky_carry_their_spin_weights(sky):
    expected = {"i": 0, "p": 2, "eth_i": 1, "eth_bar_i": -1, "eth_p": 3, "eth_bar_p": 1}
    for name, spin in expected.items():
        assert spin_weight(getattr(sky, name)) == spin, name
    for key in SECOND_ORDER_KEYS:
        assert spin_weight(sky.second_order[key]) == SECOND_ORDER_SPINS[key], key
    # Q and U taken separately have no spin weight
    assert spin_weight(sky.q) is None and spin_weight(sky.u) is None


def test_derivatives_computed_by_the_sky_carry_their_spin_weights():
    rng = np.random.default_rng(3)
    sky = Sky.from_maps(*rng.normal(size=(3, 12 * 4**2)), lmax=8)
    assert [spin_weight(m) for m in (sky.eth_i, sky.eth_p, sky.eth_bar_p)] == [1, 3, 1]
    assert spin_weight(sky.second_order["eth2_p"]) == 4


def test_arithmetic_follows_the_spin_weights():
    a, b = SpinMap(np.ones(NPIX), 1), SpinMap(np.ones(NPIX, complex), 2)
    assert (a * b).spin == 3
    assert (b / a).spin == 1
    assert (0.5j * np.exp(0.3j) * a).spin == 1
    assert np.conj(b).spin == -2 and b.conj().spin == -2
    assert np.abs(b).spin == 0
    assert (b**2).spin == 4 and np.square(a).spin == 2
    assert (a + a).spin == 1 and (-a).spin == 1
    assert (0 + a).spin == 1 and sum([a, a]).spin == 1  # adding zero is allowed
    assert a[:5].spin == 1 and a.copy().spin == 1 and a.astype(complex).spin == 1
    # real and imaginary parts of a spin-weighted map have no spin weight
    assert not isinstance(b.real, SpinMap) and not isinstance(np.imag(b), SpinMap)
    assert np.real(SpinMap(np.ones(NPIX), 0)).spin == 0
    # an array without spin weight makes the result unknown
    assert spin_weight(a * np.ones(NPIX)) is None
    assert spin_weight(a + np.ones(NPIX)) is None
    assert spin_weight(np.exp(b)) is None and np.exp(SpinMap(np.ones(3), 0)).spin == 0
    # other NumPy functions, reductions and comparisons give plain values
    assert type(np.stack([a, a])) is np.ndarray
    assert type(a > 0) is np.ndarray and not isinstance(a.sum(), SpinMap)


def test_in_place_operations_update_the_spin_weight():
    a = SpinMap(np.ones(NPIX, complex), 1)
    a *= SpinMap(np.ones(NPIX), 2)
    assert a.spin == 3
    plain = np.zeros(NPIX, complex)
    plain += a  # a plain array stays plain
    assert type(plain) is np.ndarray


def test_adding_maps_of_different_spin_weights_raises(sky):
    with pytest.raises(SpinWeightError, match="spin weights 1 and 3"):
        sky.eth_i + sky.eth_p
    with pytest.raises(SpinWeightError):
        sky.p - 1.0  # a constant has spin weight 0


def test_a_component_at_the_wrong_couple_raises(sky):
    rho, chi = 1e-3, 0.4
    with pytest.raises(SpinWeightError, match=r"S_\{1,0\} has spin weight 3"):
        SignalModel({(1, 0): -rho / 4.0 * np.exp(1j * chi) * sky.eth_p})
    with pytest.raises(SpinWeightError, match="spin weight -1"):
        SignalModel({(1, 0): np.conj(sky.eth_i)})  # the conjugate sits at (-1, 0)
    # the conjugate couple and plain arrays are accepted
    SignalModel({(-1, 0): np.conj(sky.eth_i), (2, 0): np.ones(sky.npix, complex)})


def test_models_store_plain_arrays(sky):
    model = sy.sky_signal(sky, hwp=True) * sy.sky_signal(sky, hwp=True)
    assert all(type(c.map) is np.ndarray for c in model.components())


def test_a_sky_rejects_maps_of_another_spin_weight(sky):
    with pytest.raises(SpinWeightError, match="eth_p has spin weight 3"):
        Sky(sky.i, sky.p, eth_i=sky.eth_i, eth_p=sky.eth_bar_p, eth_bar_p=sky.eth_bar_p)
    Sky(sky.i, sky.p, eth_i=np.asarray(sky.eth_i))  # plain arrays get their spin weight


def test_spin_maps_survive_pickling():
    a = pickle.loads(pickle.dumps(SpinMap(np.arange(3.0), 2)))
    assert a.spin == 2 and np.array_equal(a, np.arange(3.0))


#: Systematics whose components are not products of the maps of a Sky.
UNCHECKED = {
    # its Mueller chain combines I, Q and U, and Q and U alone have no spin weight
    "hwp_mueller",
    # its components are spin transforms of the beam-convolved coefficients
    "beam",
}


def test_every_systematic_builds_components_with_spin_weights(monkeypatch):
    built: dict[str, list] = {}
    init = SignalModel.__init__
    names = {f.__name__ for f in sy.ALL}

    def recording(self, components):
        # the outermost registered function of sbm.systematics in the call stack
        # (abs_pointing and hwp_wedge build their models in a shared helper)
        owner, frame = None, sys._getframe(1)
        while frame is not None and frame.f_globals.get("__name__", "").startswith(
            "sbm.systematics"
        ):
            if frame.f_code.co_name in names:
                owner = frame.f_code.co_name
            frame = frame.f_back
        if owner is not None:
            built.setdefault(owner, []).extend(
                (couple, spin_weight(value)) for couple, value in components.items()
            )
        init(self, components)

    monkeypatch.setattr(SignalModel, "__init__", recording)
    cases()  # every case of tests/test_systematics.py, built again
    for f in sy.ALL:
        name = f.__name__
        assert name in built, f"no case of {name} built a model"
        if name in UNCHECKED:
            continue
        unknown = [couple for couple, spin in built[name] if spin is None]
        assert not unknown, (
            f"{name} builds the components {unknown} from arrays without spin "
            "weight: use the maps of the Sky, or add it to UNCHECKED with the reason"
        )
