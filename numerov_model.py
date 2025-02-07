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
                 n: int,
                 l: np.ndarray[int]=None):
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
        self.n = n
        self.l = l or np.arange(n, dtype=int)
        self.l = self.l.reshape(1,1,-1)

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
        r = r[:, np.newaxis, np.newaxis]
        E = E.reshape(1, E.shape[0], -1)
        return (2 * self.particle.m * (E-self.V(r)) / const.hbar**2 - (self.l * (self.l+1) / r**2)).to('fm-2')


    def u(self,
            r: np.ndarray[float],
            E: np.ndarray[float],
            range=progress_bar_range
            ) -> Quantity["fm"]:
        """
        Numerov method to solve the equation for the wave function

        r: np.ndarray[float]
            Array of distances in fm
            shape: (N,)
        E: np.ndarray[float]
            Array of energies in MeV
            shape: (M,) or (M, L)
        range: 
            the range method to use (can be range or tqdm, or any other method that can iterate over a range)
        """
        r = r * self.particle.a_B
        dr = r[1] - r[0]
        u = np.zeros((r.size, E.shape[0], self.l.size))
        E = E * self.particle.R_y
        u[1] = dr.value**(self.l[0]+1)
        w = self.W(E, r)
        w = dr**2 * w / 12
        for i in range(2, len(r)):
            u[i, :] = ((2 - w[i-1,:] * 10) * u[i-1,:] - (1 + w[i-2,:]) * u[i-2,:]) / (1 + w[i,:]).value
        return u / np.sqrt(np.trapezoid(u**2, r, axis=0)).value
    
    def find_root(self, E_max: float, E_min: float, r: np.ndarray[float], N: int = 10):
        """
        find the eneergy where the wave function u(R,E) = 0

        E_max: float | np.ndarray[float]
            Maximum energy to search for the root as a multiple of the Rydberg energy
            if array, shape: (L,) where L is the number of angular momentum quantum numbers
        E_min: float | np.ndarray[float]
            Minimum energy to search for the root as a multiple of the Rydberg energy
            if array, shape: (L,) where L is the number of angular momentum quantum numbers
        r: np.ndarray[float]
            Array of distances in fm
            shape: (M,)
        N: int (optional)
            Number of devisions to make in the search for the root
        """
        E_list = []
        self.find_root_helper(E_list, E_max*np.ones(self.l.size), E_min*np.ones(self.l.size), r, N)
        df = DataFrame()
        df['l'] = [l for l, _ in E_list]
        df['E'] = [E for _, E in E_list]
        return df
        


    def find_root_helper(self,
                         E_list: list[tuple[int, np.ndarray[float]]],
                         E_max: np.ndarray[float],
                         E_min: np.ndarray[float],
                         r: np.ndarray[float],
                         N: int = 10,
                         u_min: np.ndarray[float] = None,
                         u_max: np.ndarray[float] = None
                         ) -> None:
        """
        find the eneergy where the wave function u(R,E) = 0

        E_max: float | np.ndarray[float]
            Maximum energy to search for the root as a multiple of the Rydberg energy
            if array, shape: (L,) where L is the number of angular momentum quantum numbers
        E_min: float | np.ndarray[float]
            Minimum energy to search for the root as a multiple of the Rydberg energy
            if array, shape: (L,) where L is the number of angular momentum quantum numbers
        r: np.ndarray[float]
            Array of distances in fm
            shape: (M,)
        N: int (optional)
            Number of devisions to make in the search for the root

        u_min: np.ndarray[float] (optional)
            wave function at the minimum energy
            shape: (L,)

        u_max: np.ndarray[float] (optional)
            wave function at the maximum energy
            shape: (L,)

        returns: list[tuple[int, np.ndarray[float]]]
            List of tuples with the angular momentum quantum number and the energy where the wave function is zero
        """
        mid = (E_max + E_min) / 2
        for l in self.l.flatten():
            if E_max[l] != np.nan and (mid[l] == E_max[l] or mid[l] == E_min[l]):
                con = abs(u_min[l]) < abs(u_max[l])
                E_list.append((l, E_min[l] * con + E_max[l] * ~con))
                E_max[l] = E_min[l] = np.nan
        if np.isnan(E_max).all():
            print(flush=True)
            return
        print(f"diff={abs(E_max - E_min)}, {u_min=}, {u_max=}", end='\r', flush=True)
        E_bounds = np.linspace(E_min, E_max, N + 1)
        # if len(E_bounds.shape) == 1:
        #     E_bounds = E_bounds.reshape(-1, 1) * np.ones(self.l.size)
        u = self.u(r, E_bounds, range=range)[-1]
        # u shape: (N, L)
        # u_min, u_max shape: (L,)
        for i in range(N):
            if np.any(u[i]*u[i+1]<0):
                self.find_root_helper(E_list, E_bounds[i], E_bounds[i+1], r, N, u[i], u[i+1])
        
    
    def relative_error(self, E: float, n: int=None) -> float:
        """
        relative error of the energy
        """
        n = n or self.n
        return abs(1 + n**2 * E)


    @property
    def a_B(self):
        return self.particle.a_B
    
    @property
    def R_y(self):
        return self.particle.R_y
    

    def __str__(self):
        return f"particle {self.particle.name} in state |n={self.n},l={self.l}⟩"
