from utils import const, progress_bar_range
from astropy.units.quantity import Quantity
import numpy as np
from typing import Callable
from classes import Particle
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
                 p: Particle,
                 V: Callable[[np.ndarray[Quantity["fm"]]], np.ndarray[Quantity["MeV"]]],
                 l: int):
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
        self.particle = p
        self.V = V
        self.l = l

    def W(self,
          E: np.ndarray[Quantity["MeV"]],
          r: np.ndarray[Quantity["fm"]]
          ) -> np.ndarray[Quantity["fm-2"]]:
        """
        the W function for the given energy and distance

        E: np.ndarray[Quantity["MeV"]]
            Energy of the particle
            shape: (M,) or (M, L)
        r: np.ndarray[Quantity["fm"]]
            Array of distances
            shape: (N,)
        
        returns: np.ndarray[Quantity["fm-2"]]
            W function
            shape: (N, M, L)
            where L is the number of angular momentum quantum numbers
        """
        r = r[:, np.newaxis]
        # E = E.reshape(1, E.shape[0], -1)
        return (2 * self.particle.m * (E-self.V(r)) / const.hbar**2 - (self.l * (self.l+1) / r**2)).to('fm-2')


    def u(self,
            r: np.ndarray[float],
            E: np.ndarray[float],
            range=progress_bar_range
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
        r = r * self.particle.a_B
        dr = r[1] - r[0]
        if np.isscalar(E):
            E = np.array([E])
        u = np.zeros((r.size, E.shape[0]))
        E = E * self.particle.R_y
        u[1] = dr.value**(self.l+1)
        w = self.W(E, r)
        w = dr**2 * w / 12
        for i in range(2, len(r)):
            u[i, :] = ((2 - w[i-1,:] * 10) * u[i-1,:] - (1 + w[i-2,:]) * u[i-2,:]) / (1 + w[i,:]).value
        return (u / np.sqrt(np.trapezoid(u**2, r, axis=0)).value).squeeze()
    
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
        
    
    def relative_error(self, E: float, n: int) -> float:
        """
        relative error of the energy
        """
        return abs(1 + n**2 * E)

    @property
    def a_B(self):
        return self.particle.a_B
    
    @property
    def R_y(self):
        return self.particle.R_y
    
    def __str__(self):
        return f"particle {self.particle.name} in state |l={self.l}⟩"
