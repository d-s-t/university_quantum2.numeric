from astropy import units
from astropy.units.quantity import Quantity
import astropy.constants as const
import functools
import operator

units.MeVc2 = units.def_unit('MeV / c2', units.MeV/const.c**2, format={'latex': r'\mathrm{MeV}/c^2'})

const.nuclee_mass_estimation = 931.49432 * units.MeVc2

const.hbarc = (const.hbar * const.c).to(units.MeV * units.fm)


def to_latex(q: Quantity, presision=3):
    return q.to_string(format='latex', precision=presision).replace("$", "")

def flatten(a):
    return functools.reduce(operator.iconcat, a, [])