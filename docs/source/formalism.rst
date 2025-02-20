Formalism
=========

The case of single detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SBM addresses the map-making problem in spin space. By this approach, binnning of time-ordered data (TOD)
will be replaced to a convolution of the map in spin space.

The signal field :math:`S` (:class:`.SignalFields`) is defined as a function of the detector's crossing angle :math:`\psi`
and the HWP angle :math:`\phi`. The real space scan field :math:`h` is also a function of
:math:`\Omega`, :math:`\psi`, and :math:`\phi`. The signal detected by a detector within a sky pixel
of spherical coordinates :math:`\Omega=(\theta, \varphi)` is given by

.. math::

    S^{d}(\Omega,\psi,\phi)=h(\Omega,\psi,\phi)S(\Omega,\psi,\phi).

Since the signal field is expanded to a two dimensional field given by :math:`\psi` and
:math:`\phi`, we consider corresponding scan field :math:`h` as

.. math::

    h(\Omega,\psi, \phi)= \frac{4 \pi^{2}}{N_{\rm hits}(\Omega)}\sum_{j}\delta(\psi
    -\psi_{j})\delta(\phi-\phi_{j}).


Now we consider Fourier transform to bring the signal field to spin space. Defining
:math:`n` and :math:`m` as the spin moment that is the variable conjugate to the angle :math:`\psi`
and :math:`\phi`, the transformation :math:`(\psi,\phi)\to(n,m)` is given by

.. math::

    S^{d}(\Omega,\psi,\phi) = \sum_{n,m}{}_{n,m}\tilde{S}^{d}(\Omega)e^{i n\psi}e^{i m\phi}, \\
    {}_{n,m}\tilde{S}^{d}(\Omega) = \sum_{n'=-\infty}^{\infty}\sum_{m'=-\infty}^{\infty}{}_{\Delta n,\Delta m}\tilde{h}(\Omega){}_{n',m'}\tilde{S}(\Omega),

where we introduce :math:`\Delta n = n-n'` and :math:`\Delta m = m-m'`. :math:`{}_{n,m}\tilde{S}^{d}` can be obtained :meth:`.SignalFields.get_coupled_field()`.
And define the two dimensional orientation function, :math:`{}_{\Delta n,\Delta m}\tilde{h}` by Fourier transform of the real
space scan field as

.. math::

    {}_{n,m}\tilde{h}(\Omega) &= \frac{1}{4\pi^{2}}\int d\psi \int d\phi h(\Omega,\psi,\phi)e^{-i n\psi}e^{-i m\phi} \\
    &= \frac{1}{N_{\rm hits}}\sum_{j}e^{-i(n\psi_j + m \phi_j)}.

This is waht we refer to as the cross-link which can be obtained by :meth:`.ScanFields.get_xlink()`.



The case of multiple detectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nowadays, the CMB experiment usually has multiple detectors about :math:`10^{3}` to :math:`10^{4}` to take statistics. Here we consider the implementation of multiple detectors in the map-making procedure. This can be simply described by modifying several quantities in the previous section. We introduce the detector index :math:`\mu` and total number of detectors :math:`N_{\rm dets}`, then the total number of hits per pixel :math:`N_{\rm tot}` is given by

.. math::

    N_{\rm tot}(\Omega) = \sum_{\mu}N_{\rm hits}^{(\mu)}(\Omega),

where :math:`N_{\rm hits}^{(\mu)}` is the number of hits of the :math:`\mu^{\rm th}` detector in the sky pixel :math:`\Omega`. The orientation function given by total number of observations, :math:`{}_{\Delta n,\Delta m}\tilde{h}_{\rm tot}`, is

.. math::

    {}_{n,m}h_{\rm tot}(\Omega) = \frac{1}{N_{\rm tot}(\Omega)}\sum_{\mu}{}_{n,m}\tilde{h}^{(\mu)}(\Omega)N_{\rm hits}^{(\mu)}(\Omega).

Here, we define orthogonal pair detector which is named as :math:`\texttt{T}` and :math:`\texttt{B}` that stands for *Top* and *Bottom* detectors. These detectors observe the same direction though different crossing angle :math:`\psi`, let us denote the crossing angle of the :math:`\texttt{T}` and :math:`\texttt{B}` detectors as :math:`\psi^{\texttt{T}}` and :math:`\psi^{\texttt{B}}`, respectively. Then, the orientation function of the :math:`\texttt{T}` can be exchanged to that of the :math:`\texttt{B}` by the following relation

.. math::

    {}_{n,m}\tilde{h}^{(\texttt{B})} = {}_{n,m}\tilde{h}^{(\texttt{T})}e^{i n \frac{\pi}{2}}.

The detected signal in spin space per detector is given by

.. math::

    {}_{n,m}{S^{d}}^{(\mu)}(\Omega) = \sum_{n'=-\infty}^{\infty}\sum_{m'=-\infty}^{\infty}{}_{n-n',m-m'}\tilde{h}^{(\mu)}(\Omega) {}_{n',m'}\tilde{S}^{(\mu)}(\Omega).
