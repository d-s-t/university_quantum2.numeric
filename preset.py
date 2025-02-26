from classes import Nucleus, Particle, TwoBodySystem
from utils import units, const
from typing import Callable
from astropy.units.quantity import Quantity
from numpy import ndarray

H1 = Nucleus('H1', Z=1, A=1, R=0.88 * units.fm)
H2 = Nucleus('H2', Z=1, A=2, R=2.14 * units.fm)
He4 = Nucleus('He', Z=2, A=4, R=1.97 * units.fm)

K = Particle('K', m=493.7 * units.MeVc2)

H1K = H1 + K
H2K = H2 + K
He4K = He4 + K

def Vc(sys: TwoBodySystem) -> Callable[[ndarray[Quantity["fm"]]], ndarray[Quantity["MeV"]]]:
    A = -const.alpha * sys.nucleus.Z * const.hbarc
    def V(r):
        return A / r
    return V

def get_non_rel_W(sys: TwoBodySystem, l, Vc=Vc) -> Callable[[ndarray[Quantity["MeV"]],ndarray[Quantity["fm"]]], ndarray[Quantity["fm-2"]]]:
    from numpy import newaxis, isscalar
    V = Vc(sys)
    L = l*(l+1)
    A = 2 * sys.mu / const.hbar**2
    def W(E, r):
        if not isscalar(E):
            r = r[:, newaxis]
        return (A * (E-V(r)) - (L / r**2)).to('fm-2')
    return W


def get_rel_W(sys: TwoBodySystem, l, Vc=Vc) -> Callable[[ndarray[Quantity["MeV"]],ndarray[Quantity["fm"]]], ndarray[Quantity["fm-2"]]]:
    from numpy import newaxis, isscalar
    V = Vc(sys)
    L = l*(l+1)
    Em = sys.mu * const.c**2
    def W(E, r):
        if not isscalar(E):
            r = r[:, newaxis]
        return (((E + Em - V(r))**2 - Em**2)/(const.hbarc**2) - (L / r**2)).to('fm-2')
    return W


