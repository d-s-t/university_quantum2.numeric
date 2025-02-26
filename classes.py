from dataclasses import dataclass
from astropy.units.quantity import Quantity
from utils import const
from typing import Union

@dataclass
class Particle:
    """
    Particle
    name: str
        Name of the particle
    m: Quantity["mass"]
        Mass of the particle
    property a_B: Quantity["length"]
        Bohr radius of the particle
    property R_y: Quantity["energy"]
        Rydberg energy of the particle
    """
    name: str
    m: Quantity["mass"]
    # Z: int
    
    # @property
    # def a_B(self) -> Quantity["fm"]:
    #     return (const.hbarc / (self.m * const.alpha * self.Z * const.c**2)).to('fm')
    
    # @property
    # def R_y(self) -> Quantity["MeV"]:
    #     return (const.alpha**2 * self.Z**2 * self.m * const.c**2 / 2).to('MeV')



@dataclass
class Nucleus(Particle):
    """
    Nucleus(Particle)
    A: int
        Mass number of the nucleus (number of nucleons)
    R: Quantity["length"]
        Radius of the nucleus
    Z: int
        Charge of the particle
    """
    A: int
    R: Quantity["length"]
    Z: int

    def __init__(self, name: str, Z: int, A: int, R: Quantity["length"]):
        super().__init__(name, A * const.nuclee_mass_estimation)
        self.A = A
        self.R = R
        self.Z = Z

    def __add__(self, other: Union['Particle', 'Nucleus']) -> 'TwoBodySystem':
        return TwoBodySystem(self, other)


class TwoBodySystem:
    def __init__(self, n: Nucleus, p: Particle):
        from utils import reduced_mass
        self.nucleus = n
        self.particle = p
        self.mu = reduced_mass(n.m, p.m)
        self.a_B = (const.hbarc / (self.mu * const.alpha * n.Z * const.c**2)).to('fm')
        self.R_y = (const.alpha**2 * n.Z**2 * self.mu * const.c**2 / 2).to('MeV')
    

