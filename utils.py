from astropy import units
from astropy.units.quantity import Quantity
import astropy.constants as const
import functools
import operator
import plotly.io as pio
from plotly import graph_objects as go

units.MeVc2 = units.def_unit('MeV / c2', units.MeV/const.c**2, format={'latex': r'\mathrm{MeV}/c^2'})

const.nuclee_mass_estimation = 931.49432 * units.MeVc2

const.hbarc = (const.hbar * const.c).to(units.MeV * units.fm)

plotly_show_config = config={'toImageButtonOptions': {'format': 'svg',
                                                       'filename': 'unset',
                                                       'width': 600, 'height': 450}}
pio.templates["transparent"] = pio.templates[pio.templates.default].update(
    layout=dict(
        paper_bgcolor='rgba(0,0,0,0)'
    )
)
pio.templates.default = "transparent"


def to_latex(q: Quantity, presision=3):
    return q.to_string(format='latex', precision=presision).replace("$", "")

def flatten(a):
    return functools.reduce(operator.iconcat, a, [])