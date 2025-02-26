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
        u = u * dr.unit**-0.5
        E = E * self.R_y
        w = self.W(E, r)
        w = dr**2 * w / 12
        w1 = 1 + w
        w2 = 2 - 10 * w
        for i in range(2, len(r)):
            u[i] = (w2[i-1] * u[i-1] - w1[i-2] * u[i-2]) / w1[i]
        norm = np.sqrt(np.trapezoid(u**2, r, axis=0))
        return u / norm
    
    def find_root(self, E_max: float, E_min: float, r: np.ndarray[float], D: int = 10):
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
        E_list = []
        self.find_root_helper(E_list, E_max, E_min, r, D)
        print()
        return np.array(E_list)
        


    def find_root_helper(self,
                         E_list: list[tuple[int, np.ndarray[float]]],
                         E_max: np.ndarray[float],
                         E_min: np.ndarray[float],
                         r: np.ndarray[float],
                         D: int = 10,
                         u_min: np.ndarray[float] = -np.inf,
                         u_max: np.ndarray[float] = np.inf
                         ) -> None:
        """
        find the energy E where the wave function u(R,E) = 0

        E_max: float
            Maximum energy to search for the root as a multiple of the Rydberg energy
        E_min: float
            Minimum energy to search for the root as a multiple of the Rydberg energy
        r: np.ndarray[float]
            Array of distances in fm
            shape: (N,)
        D: int (optional)
            Number of devisions to make in the search for the root
        u_min: float
            wave function at R for the minimum energy
        u_max: float
            wave function at the maximum energy

        returns: list[float]
            list of energies where the wave function is zero at R
        """
        mid = (E_max + E_min) / 2
        if mid in (E_max, E_min):
            con = abs(u_min) < abs(u_max)
            E_list.append(E_min * con + E_max * ~con)
            return
        print(f"diff={abs(E_max - E_min):.0e}, {u_min=:.0e}, {u_max=:.0e}", end='\r', flush=True)
        E_bounds = np.linspace(E_min, E_max, D + 1)
        u = self.u(r, E_bounds, range=range)[-1]
        # u shape: (D+1,)
        for i in range(D):
            if u[i]*u[i+1]<0:
                self.find_root_helper(E_list, E_bounds[i], E_bounds[i+1], r, D, u[i], u[i+1])

    @property
    def a_B(self):
        return self.system.a_B
    
    @property
    def R_y(self):
        return self.system.R_y


