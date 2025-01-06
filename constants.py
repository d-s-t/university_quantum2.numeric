from astropy import units
import astropy.constants as const

units.MeVc2 = units.def_unit('MeV / c2', units.MeV/const.c**2, format={'latex': r'\mathrm{MeV}/c^2'})

const.nuclee_mass_estimation = 931.49432 * units.MeVc2

const.hbarc = (const.hbar * const.c).to(units.MeV * units.fm)
