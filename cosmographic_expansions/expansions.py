import numpy as np
from astropy import constants as const
from typing import Tuple
from scipy import integrate

class ExpansionModels:
    SPEED_OF_LIGHT = const.c.to('km/s').value
    @staticmethod
    def _redshift_expansion(z: np.ndarray, H0: float, *a_coeffs: float, order: int = 3) -> np.ndarray:
        c = ExpansionModels.SPEED_OF_LIGHT
        # a_coeffs: a0, a1, a2, ... depending on order
        series = 0.0
        if order>6:
            raise ValueError(f"Order {order} not supported. Use order 1–6.") 
        for i, a in enumerate(a_coeffs[:order]):
            series += a * z**i
        return c * z / H0 * series


    @classmethod
    def z_series(cls, z: np.ndarray, H0: float, q0: float, j0: float = 0,
        s0: float = 0, c0: float = 0, p0: float = 0, Ok: float = 0, order: int = 3) -> np.ndarray:
        # compute coefficients (a0..a5) same as original file; here we compute a subset for readability
        
        a0 = 1.0
        a1 = 1.0/2.0 - q0/2.0
        a2 = -1.0/6.0 - j0/6.0 + q0/6.0 + q0**2/2.0 + Ok/6.0
        a3 = (1.0/12.0 + 5*j0/24.0 - q0/12.0 + 5.0*j0*q0/12.0 - 
              5.0*q0**2/8.0 - 5.0*q0**3/8.0 + s0/24.0 - Ok/12.0 - Ok*q0/4.0)
        a4 = (-1.0/20.0 - 9*j0/40.0 + j0**2/12.0 - c0/120.0 + q0/20.0 - 
              11*j0*q0/12.0 + 27*q0**2/40.0 - 7*j0*q0**2/8.0 + 11*q0**3/8.0 + 
              7*q0**4/8.0 - 11*s0/120.0 - q0*s0/8.0 - Ok**2 - 
              5*Ok*(2*j0 - 9*q0**2 - 8*q0 - 1))
        a5 = (1.0/30.0 + 7*j0/30.0 - 19.0*j0**2/72.0 + 19.0*c0/720.0 + 
              p0/720.0 - q0/30.0 + 13*j0*q0/9.0 - 7*j0**2*q0/18.0 + 
              7*c0*q0/240.0 - 7*q0**2/10.0 + 133.0*j0*q0**2/48.0 - 
              13.0*q0**3/6.0 + 7.0*j0*q0**3/4.0 - 133.0*q0**4/48.0 - 
              21.0*q0**5/16.0 + 13.0*s0/90.0 - 7.0*j0*s0/144.0 + 
              19.0*q0*s0/48.0 + 7*q0**2*s0/24.0 - 
              15.*Ok*(-14*j0*q0 - 9*j0 + 28*q0**3 + 40*q0**2 + 17*q0 - s0 + 1) - 
              Ok**2*(15*q0 + 9))
        return cls._redshift_expansion(z, H0, a0, a1, a2, a3, a4, a5, order=order)


    @classmethod
    def y_series(cls, y: np.ndarray, H0: float, q0: float, j0: float = 0,
        s0: float = 0, c0: float = 0, p0: float = 0, Ok: float = 0, order: int = 3) -> np.ndarray:
        a0 = 1.0
        a1 = -1/2 * (q0 - 3)
        a2 = 1/6 * (11 - 5*q0 + 3*q0**2 - j0 + Ok)
        a3 = (1/24 * (50 - 7*j0 - 26*q0 + 10*q0*j0 + 21*q0**2 - 
              15*q0**3 + s0 + 10*Ok - 6*q0*Ok))
        a4 = (1/120 * (274 + 10*j0**2 - c0 - 154*q0 + 141*q0**2 - 
              135*q0**3 + 105*q0**4 + 9*s0 - 15*q0*s0 + 85*Ok - 
              80*q0*Ok + 45*q0**2*Ok + Ok**2 - 
              j0*(47 - 90*q0 + 105*q0**2 + 10*Ok)))
        a5 = (1/720 * (1764 + p0 + j0**2*(110 - 280*q0) - 1044*q0 + 
              1026*q0**2 - 1110*q0**3 + 1155*q0**4 - 945*q0**5 + 
              c0*(-11 + 21*q0) + 74*s0 - 165*q0*s0 + 210*q0**2*s0 + 
              735*Ok - 855*q0*Ok + 750*q0**2*Ok - 420*q0**3*Ok + 
              15*s0*Ok + 21*Ok**2 - 15*q0*Ok**2 + 
              j0*(-342 - 1155*q0**2 + 1260*q0**3 - 35*s0 - 165*Ok + 
              10*q0*(74 + 21*Ok))))
        return cls._redshift_expansion(y, H0, a0, a1, a2, a3, a4, a5, order=order)


    @classmethod
    def log_series(cls, log1z: np.ndarray, H0: float, q0: float, j0: float = 0,
        s0: float = 0, c0: float = 0, p0: float = 0, Ok: float = 0, order: int = 3) -> np.ndarray:
        a0 = 1.0
        a1 = 1 - 1/2*q0
        a2 = 1/6 * (3 - 2*q0 + 3*q0**2 - j0 + Ok)
        a3 = (1/6 + (1/8)*(-q0 + q0**2 - 5*q0**3) + (5/12)*q0*j0 + 
              1/24*(-j0 + s0 + 4*Ok - 6*q0*Ok))
        a4 = (1/120 * (5 + 10*j0**2 - c0 - 4*q0 + 6*q0**2 + 15*q0**3 + 
              105*q0**4 - s0 - 15*q0*s0 + 10*Ok - 20*q0*Ok + 45*q0**2*Ok + 
              Ok**2 - j0*(2 + 10*q0 + 105*q0**2 + 10*Ok)))
        a5 = (1/720 * (6 + p0 - 5*q0 + 6*q0**2 - 60*q0**3 - 420*q0**4 - 
              945*q0**5 - 40*j0**2*(1 + 7*q0) + c0*(4 + 21*q0) + 4*s0 + 
              60*q0*s0 + 210*q0**2*s0 + 20*Ok - 45*q0*Ok + 75*q0**2*Ok - 
              420*q0**3*Ok + 15*s0*Ok + 6*Ok**2 - 15*q0*Ok**2 + 
              j0*(-2 + 420*q0**2 + 1260*q0**3 - 35*s0 - 15*Ok + 
              10*q0*(4 + 21*Ok))))
        return cls._redshift_expansion(log1z, H0, a0, a1, a2, a3, a4, a5, order=order)
    


class EISModel:
    """
    EIS (Expansion in Inverse Scale factor) model for luminosity distance.
    Supports orders 1–5.
    """
    SPEED_OF_LIGHT = const.c.to('km/s').value  # km/s

    @staticmethod
    def integrand(z: float, E1: float, E2: float, E3: float, E4: float, E5: float, order: int) -> float:
        """
        Integrand for EIS expansion.
        Returns: 1 / (E(z))
        """
        E0 = 1
        if order ==1:
            return 1 / E0 
        if order == 2:
            return 1 / (E0 + E1 * z)
        elif order == 3:
            return 1 / (E0 + E1 * z + (E2 * z ** 2) / 2)
        elif order == 4:
            return 1 / (E0 + E1 * z + (E2 * z ** 2) / 2 + (E3 * z ** 3) / 6)
        elif order == 5:
            return 1 / (E0 + E1 * z + (E2 * z ** 2) / 2 + (E3 * z ** 3) / 6 + (E4 * z ** 4) / 24)
        elif order == 6:
            return 1 / (E0 + E1 * z + (E2 * z ** 2) / 2 + (E3 * z ** 3) / 6 + (E4 * z ** 4) / 24 + (E5 * z ** 5) / 120)
        else:
            raise ValueError(f"Order {order} not supported. Use order 1–6.")

    @classmethod
    def luminosity_distance(cls, z: np.ndarray, H0: float, E1: float, E2=None, E3=None, E4=None, E5=None, order: int = 2) -> np.ndarray:
        """
        Calculate luminosity distance using EIS expansion.
        Returns d_L in Mpc.
        """
        dL = np.zeros_like(z, dtype=float)
        for i, z_val in np.ndenumerate(z):
            integral, _ = integrate.quad(cls.integrand, 0, z_val, args=(E1, E2, E3, E4, E5, order))
            dL[i] = (1 + z_val) * integral
        return (cls.SPEED_OF_LIGHT / H0) * dL
    
class EISModelJAX:
    """
    JAX-compatible EIS (Expansion in Inverse Scale factor) model.
    Use this version for MCMC with JAX/NUTS samplers.
    
    For non-JAX code, use the original EISModel class.
    
    Note: JAX is imported only when this class is instantiated or used,
    so users without JAX installed won't get import errors unless they
    actually try to use this class.
    """
    SPEED_OF_LIGHT = const.c.to('km/s').value  # km/s
    _jax_imported = False
    _jnp = None
    _vmap = None
    
    @classmethod
    def _ensure_jax(cls):
        """Lazy import of JAX - only imports when actually needed"""
        if not cls._jax_imported:
            try:
                import jax.numpy as jnp
                from jax import vmap
                cls._jnp = jnp
                cls._vmap = vmap
                cls._jax_imported = True
            except ImportError:
                raise ImportError(
                    "JAX is required to use EISModelJAX. "
                    "Install it with: pip install jax jaxlib\n"
                    "For GPU support, see: https://github.com/google/jax#installation"
                )
    
    @staticmethod
    def integrand(z, E1, E2, E3, E4, E5, order):
        """
        Integrand for EIS expansion.
        Returns: 1 / E(z)
        
        Note: Fully vectorized for JAX
        """
        E0 = 1.0
        
        if order == 1:
            return 1.0 / E0
        elif order == 2:
            return 1.0 / (E0 + E1 * z)
        elif order == 3:
            return 1.0 / (E0 + E1 * z + (E2 * z ** 2) / 2)
        elif order == 4:
            return 1.0 / (E0 + E1 * z + (E2 * z ** 2) / 2 + (E3 * z ** 3) / 6)
        elif order == 5:
            return 1.0 / (E0 + E1 * z + (E2 * z ** 2) / 2 + (E3 * z ** 3) / 6 + 
                          (E4 * z ** 4) / 24)
        elif order == 6:
            return 1.0 / (E0 + E1 * z + (E2 * z ** 2) / 2 + (E3 * z ** 3) / 6 + 
                          (E4 * z ** 4) / 24 + (E5 * z ** 5) / 120)
        else:
            raise ValueError(f"Order {order} not supported. Use order 1–6.")
    
    @classmethod
    def _integrate_single_obselete(cls, z_val, E1, E2, E3, E4, E5, order):
        """
        Integrate using 32-point Gauss–Legendre quadrature.
        Much more accurate than trapezoidal rule.
        """
        jnp = cls._jnp

        # nodes and weights for n=32 Gauss–Legendre on [-1, 1]
        nodes, weights = jnp.polynomial.legendre.leggauss(32)

        # map nodes from [-1,1] to [0, z_val]
        a = 0.0
        b = z_val
        x = 0.5 * (nodes + 1.0) * (b - a) + a

        fvals = cls.integrand(x, E1, E2, E3, E4, E5, order)

        # final quadrature
        return 0.5 * (b - a) * jnp.sum(weights * fvals)
    @classmethod
    def _integrate_single(cls, z_val, E1, E2, E3, E4, E5, order):
        """
        32-point Gauss–Legendre quadrature, hardcoded nodes & weights.
        JAX-safe replacement for leggauss(32).
        """

        jnp = cls._jnp

        # 32-point Gauss–Legendre nodes (x_i) and weights (w_i)
        nodes = jnp.array([
            -0.997263861849, -0.985611511545, -0.964762255588, -0.934906075938,
            -0.896321155766, -0.849367613733, -0.794483795968, -0.732182118740,
            -0.663044266930, -0.587715757241, -0.506899908932, -0.421351276131,
            -0.331868602282, -0.239287362252, -0.144471961583, -0.048307665688,
            0.048307665688,  0.144471961583,  0.239287362252,  0.331868602282,
            0.421351276131,  0.506899908932,  0.587715757241,  0.663044266930,
            0.732182118740,  0.794483795968,  0.849367613733,  0.896321155766,
            0.934906075938,  0.964762255588,  0.985611511545,  0.997263861849
        ])

        weights = jnp.array([
            0.007018610009, 0.016274394715, 0.025392065310, 0.034273862913,
            0.042835898022, 0.050998059263, 0.058684093479, 0.065822222776,
            0.072345794109, 0.078193895787, 0.083311924227, 0.087652093004,
            0.091173878696, 0.093844399081, 0.095638720079, 0.096540088515,
            0.096540088515, 0.095638720079, 0.093844399081, 0.091173878696,
            0.087652093004, 0.083311924227, 0.078193895787, 0.072345794109,
            0.065822222776, 0.058684093479, 0.050998059263, 0.042835898022,
            0.034273862913, 0.025392065310, 0.016274394715, 0.007018610009
        ])

        # Map from [-1,1] to [0, z_val]
        a = 0.0
        b = z_val
        x = 0.5 * (nodes + 1.0) * (b - a) + a

        # Integrand evaluated at mapped nodes
        fvals = cls.integrand(x, E1, E2, E3, E4, E5, order)

        # Gauss–Legendre quadrature
        return 0.5 * (b - a) * jnp.sum(weights * fvals)

        
    @classmethod
    def luminosity_distance(cls, z, H0, E1, E2=None, E3=None, E4=None, E5=None, order=2):
        """
        Calculate luminosity distance using EIS expansion.
        
        Parameters:
        -----------
        z : jax array or array-like
            Redshift values
        H0 : float
            Hubble constant in km/s/Mpc
        E1, E2, E3, E4, E5 : float
            EIS expansion parameters
        order : int
            Expansion order (1-6)
        
        Returns:
        --------
        d_L : jax array
            Luminosity distance in Mpc
        """
        # Ensure JAX is imported
        cls._ensure_jax()
        jnp = cls._jnp
        
        # Handle None values
        E2 = E2 if E2 is not None else 0.0
        E3 = E3 if E3 is not None else 0.0
        E4 = E4 if E4 is not None else 0.0
        E5 = E5 if E5 is not None else 0.0
        
        # Convert to JAX array
        z = jnp.asarray(z)
        scalar_input = z.ndim == 0
        z = jnp.atleast_1d(z)
        
        # Vectorize integration over all redshift values
        integrate_vectorized = cls._vmap(
            lambda z_val: cls._integrate_single(z_val, E1, E2, E3, E4, E5, order)
        )
        
        integrals = integrate_vectorized(z)
        dL = (1 + z) * integrals
        result = (cls.SPEED_OF_LIGHT / H0) * dL
        
        return result[0] if scalar_input else result

