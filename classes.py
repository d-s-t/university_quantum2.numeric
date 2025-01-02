from dataclasses import dataclass
from astropy.units.quantity import Quantity
from constants import const
from typing import Union

@dataclass
class Particle:
    name: str
    m: Quantity["mass"]
    Z: int
    
    @property
    def a_B(self) -> Quantity["fm"]:
        return (const.hbarc / (self.m * const.alpha * self.Z * const.c**2)).to('fm')
    
    @property
    def R_y(self) -> Quantity["fm"]:
        return (const.alpha**2 * self.Z**2 * self.m * const.c**2 / 2).to('MeV')


    def __add__(self, other: Union['Particle', 'Attom']):
        return Particle(self.name + '+' + other.name, 1/(1/self.m + 1/other.m), self.Z + other.Z)


@dataclass
class Attom(Particle):
    A: int
    R: Quantity["length"]

    def __init__(self, name: str, Z: int, A: int, R: Quantity["length"]):
        super().__init__(name, A * const.nuclee_mass_estimation, Z)
        self.A = A
        self.R = R
