from utils import const, progress_bar_range
from astropy.units.quantity import Quantity
import numpy as np
from typing import Callable
from classes import TwoBodySystem
from pandas import DataFrame

class Numerov:
    """
    Numerov model for quantum mechanics for 2 particles for equations of the form:
    -ħ²/2μ ∇²u + V(r)u = Eu
    where it gets the form:
    d²u/dr² + [(E- V(r))(2μ/ħ²) - (l(l+1)/r²)]u = 0
    and we define the W function as:
    W(r) = (E - V(r))(2μ/ħ²) - l(l+1)/r²
    """
    def __init__(self, 
                 sys: TwoBodySystem,
                 W: Callable[[np.ndarray[float], np.ndarray[float]], np.ndarray[float]]
                 ):
        """
        p: Particle
            Particle to be used in the model
        V: Callable[[np.ndarray[Quantity["fm"]]], np.ndarray[Quantity["MeV"]]]
            Potential energy function
        n: int
            Principal quantum number
        l: np.ndarray[int] (optional)
            Azimuthal quantum number
            Defaults to np.arange(n, dtype=int)
        """
        self.W = W
        self.system = sys


    def u(self,
            r: np.ndarray[float],
            E: np.ndarray[float],
            range=progress_bar_range,
            u1: np.ndarray[float] = 1,
            ) -> np.ndarray[float]:
        """
        Numerov method to solve the equation for the wave function

        r: np.ndarray[float]
            Array of distances in fm
            shape: (N,)
        E: np.ndarray[float]
            Array of energies in MeV
            shape: (M,)
        range: 
            the range method to use (can be range or tqdm, or any other method that can iterate over a range)

        returns: np.ndarray[float]
            Array of wave functions for each energy in each distance
            shape: (N, M)
        """
        r = r * self.a_B
        dr = r[1] - r[0]
        u = np.zeros_like(r.value) if np.isscalar(E) else np.zeros((r.size, E.size))
        u[1] = u1
        E = E * self.R_y
        w = self.W(E, r)
        w = (dr**2 * w / 12).value
        w1 = 1 + w
        w2 = 2 - 10 * w
        for i in range(2, len(r)):
            u[i] = (w2[i-1] * u[i-1] - w1[i-2] * u[i-2]) / w1[i]
        norm = np.sqrt(np.trapezoid(u**2, r.value, axis=0))
        return (dr.unit**-0.5) * u / norm
    
    def find_bound_energies(self, E_max: float, E_min: float, r: np.ndarray[float], D: int = 10):
        """
        find the eneergy where the wave function u(R,E) = 0

        E_max: float
            Maximum energy to search for the root as a multiple of the Rydberg energy
        E_min: float
            Minimum energy to search for the root as a multiple of the Rydberg energy
        r: np.ndarray[float]
            Array of distances in fm
            shape: (N,)
        D: int (optional)
            Number of devisions to make in the search for the root
        """
        while True:
            E_mid = np.linspace(E_min, E_max, D + 1)
            uR_mid = self.u(r, E_mid, range=range)[-1]
            if np.any(uR_mid == 0):
                print()
                return E_mid[uR_mid == 0]
            sign_change_indices = np.where(uR_mid[1:] * uR_mid[:-1] < 0)[0]
            if len(sign_change_indices) == 0:
                print()
                raise ValueError("No root in the given range (or even number of roots)")
            idx = sign_change_indices[0]
            E_min, E_max = E_mid[idx], E_mid[idx + 1]
            u_min, u_max = uR_mid[idx], uR_mid[idx + 1]
            print(f"diff={abs(E_max - E_min):.0e}, {u_min=:.0e}, {u_max=:.0e}", end='\r', flush=True)
            if (E_max + E_min) / 2 in (E_max, E_min):
                print()
                return E_min if abs(uR_mid[idx]) < abs(uR_mid[idx + 1]) else E_max


    def find_bound_energy_alt(self, r: np.ndarray[float], E_max: float, D: int = 10):
        """
        """
        pass        

    @property
    def a_B(self):
        return self.system.a_B
    
    @property
    def R_y(self):
        return self.system.R_y
    


