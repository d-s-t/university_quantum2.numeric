from utils import const, flatten
from astropy.units.quantity import Quantity
import numpy as np
from typing import Callable
from classes import Particle
from tqdm import tqdm
from sys import float_info  

class Numerov:
    def __init__(self, p: Particle, V: Callable[[np.ndarray[Quantity["fm"]]], np.ndarray[Quantity["MeV"]]], n: int, l: int):
        self.particle = p
        self.V = V
        self.n = n
        self.l = l

    def W(self,
          E: Quantity["MeV"],
          r: np.ndarray[Quantity["fm"]]
          ) -> np.ndarray[Quantity["fm-2"]]:
        return (2 * self.particle.m * (E-self.V(r)) / const.hbar**2 - self.l * (self.l+1) / r**2).to('fm-2')


    def ure(self,
            r: np.ndarray[float],
            E: np.ndarray[float],
            progress_bar: bool = True
            ) -> Quantity["fm"]:
        r = r * self.particle.a_B
        dr = r[1] - r[0]
        u = np.zeros((*r.shape, *E.shape)) * dr.unit
        E = E * self.particle.R_y
        u[1,:] = dr
        w = dr**2 * self.W(E, r[:, np.newaxis]) / 12
        interations = tqdm(range(2, len(r))) if progress_bar else range(2, len(r))
        for i in interations:
            u[i, :] = ((2 - w[i-1,:] * 10) * u[i-1,:] - (1 + w[i-2,:]) * u[i-2,:]) / (1 + w[i,:])
        return u
    
    def find_root(self, E_max: float, E_min: float,
                  r: np.ndarray[float],
                  N: int = 10,
                  ) -> np.ndarray[float]:
        E_bounds = np.linspace(E_min, E_max, N + 1)
        u = self.ure(r, E_bounds, progress_bar=False)[-1,:]
        u_min, u_max = u[0], u[-1]
        print(f"diff={abs(E_max - E_min):.3e}, {u_min=:.3e}, {u_max=:.3e}", end='\r', flush=True)
        if abs(E_max - E_min) <= float_info.epsilon:
            print()
            return [E_min] if abs(u_min) < abs(u_max) else [E_max]
        return flatten([self.find_root(E_bounds[i+1], E_bounds[i], r, N) for i in range(N) if u[i]*u[i+1]<0])
        
    
    def etta(self, E: float):
        return abs(1 + self.n**2 * E)


    @property
    def a_B(self):
        return self.particle.a_B
    
    @property
    def R_y(self):
        return self.particle.R_y
    

    def __str__(self):
        return f"particle {self.particle.name} in state |n={self.n},l={self.l}⟩"
