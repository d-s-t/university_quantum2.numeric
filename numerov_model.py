from constants import const
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
                  N: int = 1,
                  ) -> np.ndarray[float]:
        E_bounds = np.linspace(E_min, E_max, N + 1)
        u = self.ure(r, E_bounds, progress_bar=False)
        return np.array([self.energy_binary_search(E_bounds[i], E_bounds[i+1], u[-1, i], u[-1, i+1], r) for i in range(N) if u[-1, i]*u[-1, i+1]<0]).flatten()
        

    def energy_binary_search(self, E1: float, E2: float, u1: Quantity["fm"], u2: Quantity["fm"], r: np.ndarray[float]):
        if u1*u2 > 0:
            raise ValueError('u1 and u2 should have different signs')
        
        E1, E2, u1, u2 = E1.reshape(1), E2.reshape(1), u1.reshape(1), u2.reshape(1)
        
        if u1.value == 0:
            return E1
        if u2.value == 0:
            return E2

        while abs(E2 - E1) > float_info.epsilon:
            print(f"difference: {abs(E2 - E1)[0]:.3e}, u1: {u1[0]:.3e}, u2: {u2[0]:.3e}", end='\r', flush=True)
            E_mid = (E1 + E2) / 2
            u = self.ure(r, E_mid, progress_bar=False)
            u_mid = u[-1]
            if u_mid.value == 0:
                return E_mid
            if u_mid * u1 < 0:
                E2 = E_mid
                u2 = u_mid
            else:
                E1 = E_mid
                u1 = u_mid
        print()
        return (E1 if abs(u1) < abs(u2) else E2)[0]

    
    def __str__(self):
        return f"particle {self.particle.name} in state |n={self.n},l={self.l}⟩"
