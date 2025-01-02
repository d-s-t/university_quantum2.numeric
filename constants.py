from astropy import units
from scipy import constants
import astropy.constants as const
from dataclasses import dataclass
from astropy.units.quantity import Quantity

units.MeVc2 = units.def_unit('MeV / c2', units.MeV/const.c**2, format={'latex': r'\mathrm{MeV}/c^2'})

const.nuclee_mass_estimation = 931.49432 * units.MeVc2

const.hbarc = (const.hbar * const.c).to(units.MeV * units.fm)
const.m_k = 493.7 * units.MeVc2
