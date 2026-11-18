# Handbook.md — Implementation Specifications for New Systematic Effects

This handbook defines how to specify a new systematic effect in SBM, from its physical model to its public API and tests. Write a specification with the [specification template](#spec-template) for each effect before implementing it.

The handbook does not restate the conventions of SBM (the spin expansion, the angles, the positions of the components) nor the procedures documented in `docs/source/`; it refers to them. If it disagrees with [sbm/conventions.py](../sbm/conventions.py) or the code, treat those as authoritative and fix this handbook.

## 1. Setup and Execution

Unless stated otherwise, code paths and commands in this document are relative to the repository root.

Set up `.venv` and run the tests, Ruff and Sphinx as described in [CONTRIBUTING.md](../CONTRIBUTING.md). The CI workflow [test.yml](../.github/workflows/test.yml) is the reference for the supported Python and NumPy versions, and pyproject.toml for the dependencies and the version and rules of Ruff. `.venv/` is ignored by `.gitignore`.

## 2. Model Responsibilities

A normal systematic-effect function returns only the perturbation relative to an ideal detector, `delta d = d_with_error - d_ideal`, as a `SignalModel`. Keep it independent of the observing scan and map-making.

```python
from sbm import Basis, map_make
from sbm import systematics as sy

model = sy.sky_signal(sky, hwp=True) + sy.abs_pointing(
    sky, rho_rad=1e-4, chi_rad=0.3
)
basis = Basis.iqu(hwp=True).extend(sy.abs_pointing.templates)
result = map_make((hmaps, model), basis)
```

The caller prepares `sky` and `hmaps`. When adding several effects, specify the reference model assumed by each effect and the omitted cross terms. Addition alone does not generate cross terms automatically. For nonlinear responses, use products of `SignalModel` objects as needed, for example `d + beta * (d * d)`.

The beam is the exception: it returns the complete signal. Its usage is defined in the [beam-specific specifications](#beam-spec).

Put the function in the module of the stage where the effect acts; the modules are listed in [Adding a systematic](../docs/source/adding_systematics.rst) ("Where things are"). If an existing effect can express the model with an additional argument, avoid implementing the same model twice. Write upstream code and documentation in docs/source/ in English, following [CONTRIBUTING.md](../CONTRIBUTING.md).

## 3. Mathematical and Data Contracts

### 3.1 Angles and Spin Components

The expansion of the data in spin components, the crossing and HWP angles, the positions and coefficients of the ideal components, and the relation to the h-maps of litebird_sim / Falcons.jl are defined in [sbm/conventions.py](../sbm/conventions.py) and shown on [conventions.rst](../docs/source/conventions.rst). Refer to them in a specification instead of copying them, and take them from `sbm.conventions` in the code (see "Keeping the conventions in one place" in [Agent.md](Agent.md)).

`HMaps.load` / `HMaps.save` convert the convention of h-map files; do not apply the conversion again after loading. Read SBM 0.4 cross-link files with `HMaps.load_legacy`, which has its own conversion (see its docstring).

### 3.2 Arrays and Spin Weights

- Component maps must be one-dimensional arrays of shape `(npix,)`, all with the same npix. Do not pass a scalar, such as an angle parameter, as a component map. Components with incompatible shapes raise an exception.
- The `(0, 0)` component must be real. A nonzero imaginary part raises `ValueError`.
- Register only one member of each conjugate pair. Supplying the same physical component at both `(n, m)` and `(-n, -m)` adds the contributions and counts it twice.
- The spin weight of `S[n,m]` must equal n. The HWP harmonic m does not contribute to the spin weight.
- Build components from `Sky` maps (`SpinMap`) and preserve their spin weights when multiplying by coefficients. The spin weights of the maps of `Sky` are listed in conventions.rst ("Spin weights"). `SignalModel` itself accepts plain ndarrays without checking their spin weights, but tests/test_spin.py fails registered functions that build components without spin weights; the exemptions are listed with their reasons in `UNCHECKED` in that file. `sky.q` and `sky.u` also have no spin weight individually. For maps supplied from outside `Sky`, assign their spin weight with `SpinMap(arr, s)`. For example, `bandpass_mismatch` wraps temperature maps with `SpinMap(comp, 0)`.

Complex conjugation reverses the sign of the spin weight. Use covariant derivatives for polarization; do not replace them with derivatives that treat Q and U as independent scalar maps.

### 3.3 Units and Approximations

Angle argument names must end in `_rad`. Specify dimensionless quantities such as gains, temperature maps, derivatives, and beam normalization, and keep the final TOD units consistent. For a small-angle expansion, state the retained orders and omitted terms. NET is in muK sqrt(s), and the sampling rate is in Hz.

See [conventions.rst](../docs/source/conventions.rst) and [spin.py](../sbm/spin.py) for details.

## 4. Public API and Registration

Declare and register the function as described in [Adding a systematic](../docs/source/adding_systematics.rst): the meaning of `data` and `templates` in `@systematic(data=..., templates=...)`, the import in sbm/systematics/\_\_init\_\_.py, the test case and the documentation. In addition:

- Also document components that templates cannot separate, such as an error at the same component as the original polarization. If the list of templates cannot be fixed, for example because it depends on arguments, explain why it is empty and which Basis the caller should select.
- A new API's docstring must state its arguments, return value, required Sky maps, data type, error sign convention, perturbation order, and templates. Define and verify the behavior for unsupported argument combinations and invalid input shapes or ranges, including exceptions.
- Check that no existing function models the same effect (for example, `polarization_angle` in detector.py) before adding one.
- Update both the imports and `__all__` in sbm/systematics/\_\_init\_\_.py, and add the function to the function list in that module's docstring. The decorator registers a function in `ALL` when its module is imported. Therefore, functions in a module not imported by \_\_init\_\_.py are not registered by an ordinary `import sbm.systematics` alone (importing that module directly does register them). `ALL` is a registry, not a sequence that applies effects in order; its order is the order of the imports in \_\_init\_\_.py, which Ruff is configured not to sort.
- tests/test_systematics.py checks that the set of callables in `__all__` (excluding `systematic`) equals `ALL`. Keep helper functions out of `__all__`, give their names a leading `_`, and place them inside the module; `_offset_components` in pointing.py is an example. Constants (`IDEAL_HWP`, `DATA`) may be included in `__all__`.

<a id="spec-template"></a>

## 5. Specification Template for Each Effect

Fill in the physical model and independent validation equations before implementation, then append the results after implementation. Paste the specification into the description of the pull request; do not commit it to the repository.

```markdown
# <Effect name> Implementation Specification

## Purpose and Scope
- Physical effect and reference equations or literature:
- Implementation module and public function name:
- Data type (pair / single / single_hwp / any):
- Assumptions and conditions outside the supported scope:

## Parameters
| Name | Meaning, sign, and coordinate frame | Unit | Shape, allowed range, and default |
| --- | --- | --- | --- |
| ... | ... | ... | ... |

## Physical Model
- Ideal TOD d_ideal:
- TOD with the error d_with_error:
- delta d = d_with_error - d_ideal:
- Retained perturbation orders, omitted cross terms, and validity range:
- Expected result at zero error:

## Spin Decomposition
| (n, m) | Expression for S[n,m] | Required Sky maps | Spin-weight check |
| --- | --- | --- | --- |
| ... | ... | ... | Must equal n |

## API and Composition
- Signature, return value, and required derivative maps:
- Composition with sky_signal:
- Templates, absorbed orders, and components that cannot be absorbed:
- Required h-map couples and Basis:
- Behavior for invalid inputs and unsupported conditions:

## Independent Validation and Acceptance Criteria
- Real-space TOD at the same approximation order (tests/test_systematics.py):
  - Equations independent of the spin decomposition, and the angle convention of the reference TOD (as stated in the docstrings of sbm.testing and of the helpers in tests/test_systematics.py):
  - Case name, hwp flag, and main argument branches:
  - Absorbed case: parameters whose perturbation is absorbed exactly by the templates:
- Approximation truncation error (a separate test, following test_pointing_geometry.py):
  - Exact reference signal, error magnitudes to sample, and expected residual order:
- Zero-error limit, signs, and normalization:
- Tolerances: test_systematics keeps the default tolerances of assert_matches_tod. Values used in separate tests and their justification:
- Tests, public exports, docs, and CHANGELOG to update:
- Commands run, their results, and remaining limitations:
```

For a beam specification, define the return value as the complete signal and include the additional items in the next section.

<a id="beam-spec"></a>

## 6. Beam-Specific Specifications: Optical Chain Review

The optical chain review in this section is an additional requirement only when implementing or modifying a beam.

### 6.1 Response Definitions and Ordering

When designing a beam, document the relative placement of the sky, HWP, optical responses included in the beam, and detector, along with the coordinate frame used to define each. In particular, include the following in the specification.

- Optical responses already included in the input `beam_alm`, and the new effect being added.
- Whether the new effect acts before or after the HWP.
- The order of Mueller matrices and rotations, beam normalization, and detector response.
- Confirmation that the same response is not applied through both `beam_alm` and `mueller`.

The pencil-beam limit of the current implementation with HWP is the Stokes measurement written in `point_response` in [test_hwp_convolution.py](../tests/test_hwp_convolution.py) (the rotations and their order are defined there and in [hwp_beam_convolution.rst](../docs/source/hwp_beam_convolution.rst)). Matrices act from right to left. M is the Mueller matrix in the HWP's own coordinate frame. Do not arbitrarily reorder matrices or rotations. This product is a validation reference for the pencil-beam limit; it does not mean that convolution with an arbitrary asymmetric beam can be replaced by this point measurement.

The current API does not automatically construct arbitrary optical chains. If a new layout is needed, determine whether the existing assumptions can represent it, and specify any API changes and independent reference computation required.

### 6.2 Complete-Signal Composition and Input Contracts

```python
from sbm import systematics as sy

model = sy.beam(sky_alm, beam_alm, nside, spin_k=range(5), hwp=True, mueller=M)
```

`sy.beam` returns the complete signal after the beam response, so do not add `sy.sky_signal(...)` to this `model`. If only a beam perturbation is needed, define it as the difference between complete signals from the erroneous and reference beams under the same conditions.

- `sky_alm` and `beam_alm` must be TEB or TEBV coefficients with the same lmax. Prefer `SphericalHarmonics`; litebird_sim coefficients and arrays in healpy order (mmax = lmax) are also accepted. Do not confuse IQUV Mueller-matrix components with TEB coefficients.
- `mueller` must be a finite, constant, real 4 x 4 matrix, specified with `hwp=True`. The default is the ideal HWP (see the docstring of `sbm.systematics.beam`).
- `sy.hwp_mueller(sky, harmonics)` is a separate API for angle-dependent harmonic components and returns a perturbation relative to an ideal HWP. Do not treat it as the same API as the beam's `mueller` argument.
- Asymmetric beams may require odd moments. Do not assume that the default `spin_k` is sufficient; select the required moments and upper limit. The `range(5)` above is an example covering 0–4, not a guarantee of a sufficient upper limit in general.
- Supply the h-maps in `model.required_couples(basis.full)` and the differences between Basis couples needed for the orientation matrix. Include the HWP moments produced by a general Mueller matrix. `sbm.testing.assert_matches_tod` provides an example of constructing this set of couples.

### 6.3 Approximations and Additional Beam Validation

Distinguish convolution without HWP from the default shifted narrow-beam approximation with HWP. The relation between the HWP harmonic and the order of the input beam is given in the docstring of [beam.py](../sbm/beam.py) and in hwp_beam_convolution.rst. Do not add an extra complex conjugation to the internal `C = T M T†`.

`low_ell="beam_sum"` assumes a Mueller matrix uniform in the beam-centered Ludwig-3 frame and replaces the I-input ell=0,1 2f terms arising from M_QI / M_UI with a beam multipole sum. It requires `hwp=True`. It does not make all channels or broad beams exact. See [hwp_beam_convolution.rst](../docs/source/hwp_beam_convolution.rst) for details.

When modifying a beam, validate against the pencil-beam Stokes matrix product, ideal-HWP limit, identity Mueller matrix, and an independent Wigner sum for asymmetric beams. For changes involving low ell or offsets, also check the approximation's validity range and convergence.

```sh
.venv/bin/python -m pytest -q tests/test_hwp_convolution.py tests/test_beam_low_ell.py tests/test_beam_pointing.py
```

## 7. Tests and Completion Criteria

How to add a test case is described in [Adding a systematic](../docs/source/adding_systematics.rst) ("Step 3: the test"), and what the existing tests check in [Testing](../docs/source/testing.rst). Testing has two layers. Provide both for a new effect; an effect without an approximation needs only the first.

### 7.1 Agreement at the Same Approximation Order

The case in `cases()` of tests/test_systematics.py compares `map_make` of the complete model with a least-squares fit of a TOD written directly in real space. Keep the default tolerances of `assert_matches_tod` for this layer: even for an approximate model, truncating the reference TOD at the same order makes agreement possible within numerical roundoff. Assess the approximation's validity in Section 7.2 instead of loosening these tolerances.

- Write the expected TOD directly from the physical model in real space, without reconstructing it from the implementation's spin components. Use `sbm.testing.RealSky` (random values and derivatives in each pixel, and a detector sampling an offset pointing through a Taylor expansion) or the helpers of tests/test_systematics.py written in litebird_sim's angles (`lbs_polarizer_tod`, `lbs_mueller_tod`), which keep the reference independent of SBM's angle convention as well. Their docstrings state their angle conventions.
- A case with `hwp=False` is also checked for the absence of HWP harmonics (all m = 0).
- Give each main public branch its own case: single detector, pair, HWP on/off, `second_order`, and so on. If a case needs additional random numbers, use `np.random.default_rng` with a fixed seed; the `bandpass_mismatch` case is an example.
- Verify that an ordinary perturbation vanishes at zero error and that the composed signal returns to the reference signal.
- If templates are declared, provide `absorbed=lambda: (base, base + perturbation)` with parameters whose perturbation the templates absorb exactly, such as a first-order perturbation with `second_order=False`. `test_templates_absorb_the_first_order_perturbation` checks that the polarization does not change.
- For new templates, consider adding a Basis containing them to `HWP_BASES` or `NO_HWP_BASES` so that TOD agreement is checked in that Basis for all cases as well.
- Check spin weights (Section 3.2). If an entry in `UNCHECKED` in tests/test_spin.py is needed, document the reason and the alternative independent validation.

### 7.2 Approximation Truncation Error

For an approximation such as a small-angle expansion, compare against the exact signal in a separate test and verify that the residual decreases at the next order beyond those retained as the error magnitude is varied. [test_pointing_geometry.py](../tests/test_pointing_geometry.py) demonstrates how to construct an exact reference signal (a band-limited sky sampled exactly at the displaced position) and check that the residual falls as the model is improved from sky-only to first and then second order. It uses a single error amplitude and does not estimate the orders directly; when validating the convergence order of a new effect, also compare several error magnitudes. Detect incorrect directions, signs, weights, or missing transport of the polarization frame through the residual hierarchy or convergence order.

### 7.3 Completion Checklist

- [ ] Write the effect's specification, equations, units, signs, and approximation range (in the pull request).
- [ ] Update the function, decorator, public imports, `__all__`, and the docstring in \_\_init\_\_.py.
- [ ] Add real-space TOD comparisons at the same order, the required limit and branch cases, and an absorbed case if templates are declared. For approximations, add a convergence test for the truncation error.
- [ ] Update the table in docs/source/systematics.rst, docs/source/reference.rst, and CHANGELOG.md.
- [ ] Run the relevant tests and the full test suite, and record the results.
- [ ] Check that Ruff reports no diagnostics and that Sphinx passes with `-W`.

```sh
.venv/bin/python -m pytest tests -q
.venv/bin/ruff check sbm tests docs/examples
.venv/bin/python -m sphinx -b html -W docs/source docs/build/html
```

## 8. Reference Material

- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [adding_systematics.rst](../docs/source/adding_systematics.rst)
- [conventions.rst](../docs/source/conventions.rst)
- [testing.rst](../docs/source/testing.rst)
- [hwp_beam_convolution.rst](../docs/source/hwp_beam_convolution.rst)
- [thesis_systematics.rst](../docs/source/thesis_systematics.rst) (maps the thesis equations to the code; use it for the specification's reference equations and literature)
- [adding_a_systematic.ipynb](../notebooks/adding_a_systematic.ipynb)
