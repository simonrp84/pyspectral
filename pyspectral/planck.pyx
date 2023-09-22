#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013-2023 Pytroll developers
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Cython versions of the planck radiation equation."""

import logging

cimport numpy as np
import numpy as np

import cython

try:
    import dask.array as da
except ImportError:
    da = np

LOG = logging.getLogger(__name__)

cdef float H_PLANCK = 6.62606957 * 1e-34  # SI-unit = [J*s]
cdef float K_BOLTZMANN = 1.3806488 * 1e-23  # SI-unit = [J/K]
cdef float C_SPEED = 2.99792458 * 1e8  # SI-unit = [m/s]

cdef float EPSILON = 0.000001


def c_planck(wave, temperature, wavelength=True):
    """Derive the Planck radiation as a function of wavelength or wavenumber.

    SI units.
    _planck(wave, temperature, wavelength=True)
    wave = Wavelength/wavenumber or a sequence of wavelengths/wavenumbers (m or m^-1)
    temp = Temperature (scalar) or a sequence of temperatures (K)


    Output: Wavelength space: The spectral radiance per meter (not micron!)
            Unit = W/m^2 sr^-1 m^-1

            Wavenumber space: The spectral radiance in Watts per square meter
            per steradian per m-1:
            Unit = W/m^2 sr^-1 (m^-1)^-1 = W/m sr^-1

            Converting from SI units to mW/m^2 sr^-1 (cm^-1)^-1:
            1.0 W/m^2 sr^-1 (m^-1)^-1 = 1.0e5 mW/m^2 sr^-1 (cm^-1)^-1

    """
    print(c_planck)
    units = ['wavelengths', 'wavenumbers']
    if wavelength:
        LOG.debug("Using {0} when calculating the Blackbody radiance".format(
            units[(wavelength is True) - 1]))

    if np.isscalar(temperature):
        temperature = np.array([temperature, ], dtype='float64')
    elif isinstance(temperature, (list, tuple)):
        temperature = np.array(temperature, dtype='float64')

    shape = temperature.shape
    if np.isscalar(wave):
        wln = np.array([wave, ], dtype='float64')
    else:
        wln = np.array(wave, dtype='float64')

    if wavelength:
        const = 2 * H_PLANCK * C_SPEED ** 2
        nom = const / wln ** 5
        arg1 = H_PLANCK * C_SPEED / (K_BOLTZMANN * wln)
    else:
        nom = 2 * H_PLANCK * (C_SPEED ** 2) * (wln ** 3)
        arg1 = H_PLANCK * C_SPEED * wln / K_BOLTZMANN

    with np.errstate(divide='ignore', invalid='ignore'):
        # use dask functions when needed
        np_ = np if isinstance(temperature, np.ndarray) else da
        arg2 = np_.where(np_.greater(np.abs(temperature), EPSILON),
                         np_.divide(1., temperature), np.nan).reshape(-1, 1)

    if isinstance(arg2, np.ndarray):
        # don't compute min/max if we have dask arrays
        LOG.debug("Max and min - arg1: %s  %s",
                  str(np.nanmax(arg1)), str(np.nanmin(arg1)))
        LOG.debug("Max and min - arg2: %s  %s",
                  str(np.nanmax(arg2)), str(np.nanmin(arg2)))

    try:
        exp_arg = np.multiply(arg1.astype('float64'), arg2.astype('float64'))
    except MemoryError:
        LOG.warning(("Dimensions used in numpy.multiply probably reached "
                     "limit!\n"
                     "Make sure the Radiance<->Tb table has been created "
                     "and try running again"))
        raise

    if isinstance(exp_arg, np.ndarray) and exp_arg.min() < 0:
        LOG.debug("Max and min before exp: %s  %s",
                  str(exp_arg.max()), str(exp_arg.min()))
        LOG.warning("Something is fishy: \n" +
                    "\tDenominator might be zero or negative in radiance derivation:")
        dubious = np.where(exp_arg < 0)[0]
        LOG.warning(
            "Number of items having dubious values: " + str(dubious.shape[0]))

    with np.errstate(over='ignore'):
        denom = np.exp(exp_arg) - 1
        rad = nom / denom
        radshape = rad.shape
        if wln.shape[0] == 1:
            if temperature.shape[0] == 1:
                return rad[0, 0]
            else:
                return rad[:, 0].reshape(shape)
        else:
            if temperature.shape[0] == 1:
                return rad[0, :]
            else:
                if len(shape) == 1:
                    return rad.reshape((shape[0], radshape[1]))
                else:
                    return rad.reshape((shape[0], shape[1], radshape[1]))