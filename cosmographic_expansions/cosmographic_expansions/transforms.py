from typing import Tuple, List
import numpy as np


class CosmographicTransforms:
    @staticmethod
    def to_cosmographic(H0: float, Om: float, Ok: float, w: float) -> Tuple[float, ...]:
        q0 = 0.5 * (1.0 - Ok - 3.0 * (-1.0 + Om + Ok) * w)
        j0 = 1 - Ok + (9/2) * (1.0 - Om - Ok) * w * (1 + w)


        s0 = (-7/2 + 4*Ok - Ok**2/2 + (-81/4 + 15*Ok/4) * (1 - Ok - Om) * w +
        (9*Ok/4 - 117/4 - 27*(1 - Ok - Om)/4) * (1 - Ok - Om) * w**2 -
        (27/4) * (3 - Ok - Om) * (1 - Ok - Om) * w**3)


        c0 = (35/2 - 23*Ok + 11*Ok**2/2 + (489/4 - 189*Ok/4) * (1 - Ok - Om) * w +
        (207 + 189*(1 - Ok - Om)/2 - 99*Ok/2) * (1 - Ok - Om) * w**2 +
        (621/4 + 162*(1 - Ok - Om) - 81/4*Ok) * (1 - Ok - Om) * w**3 +
        81/2 * (1 + 2*(1 - Ok - Om)) * (1 - Ok - Om) * w**4)


        p0 = (1/4 * (-455 + 681*Ok - 237*Ok**2 + 11*Ok**3) + 
              (-7407/8 + 2187*Ok/4 - 255*Ok**2/8) * (1.0 - Om - Ok) * w +
              (-6849/4 + 1449*Ok/2 - 99*Ok**2/4 - 9315*(1.0 - Om - Ok)/8 + 
               945*Ok*(1.0 - Om - Ok)/8) * (1.0 - Om - Ok) * w**2 +
              (-13041/8 + 1971*Ok/4 - 81*Ok**2/8 - 5103*(1.0 - Om - Ok)/2 + 
               621*Ok*(1.0 - Om - Ok)/4 - 567*(1.0 - Om - Ok)**2/4) * (1.0 - Om - Ok) * w**3 +
              (-729 + 243*Ok/2 - 17577*(1.0 - Om - Ok)/8 + 
               567*Ok*(1.0 - Om - Ok)/8 - 243*(1.0 - Om - Ok)**2) * (1.0 - Om - Ok) * w**4 -
              243/4 * (2 + 11*(1.0 - Om - Ok) + 2*(1.0 - Om - Ok)**2) * (1.0 - Om - Ok) * w**5)


        return H0, q0, j0, s0, c0, p0
    

class EISTransforms:
    """
    Transformations between EIS parameters and cosmographic/statefinder parameters.
    """

    @staticmethod
    def cosmographic_to_eis(q0: float, j0: float, s0: float, c0: float, p0: float, Ok: float = 0.0) -> Tuple[float, ...]:
        """Convert cosmographic parameters to EIS expansion coefficients."""
        E1 = q0 + 1
        E2 = (-Ok/2 + j0/2 - q0**2/2) * 2
        E3 = (-2*j0*q0/3 - j0/2 + q0**3/2 + q0**2/2 - s0/6) * 6
        E4 = (21*Ok**2/4 + 99*Ok*j0/2 - 223*Ok*q0**2 - 793*Ok*q0/4 - 99*Ok/4 +
              c0/24 - j0**2/6 + 25*j0*q0**2/24 + 4*j0*q0/3 + j0/2 -
              5*q0**4/8 - q0**3 - q0**2/2 + 7*q0*s0/24 + s0/3) * 24
        E5 = (399*Ok**2*q0/4 + 231*Ok**2/4 - 6953*Ok*j0*q0/6 - 769*Ok*j0 +
              2070*Ok*q0**3 + 6039*Ok*q0**2/2 + 2639*Ok*q0/2 - 539*Ok*s0/6 + 70*Ok -
              11*c0*q0/120 - c0/8 + 7*j0**2*q0/12 + j0**2/2 - 7*j0*q0**3/4 -
              25*j0*q0**2/8 - 2*j0*q0 + j0*s0/8 - j0/2 - p0/120 + 7*q0**5/8 +
              15*q0**4/8 + 3*q0**3/2 - q0**2*s0/2 + q0**2/2 - 7*q0*s0/8 - s0/2) * 120
        return E1, E2, E3, E4, E5
    @staticmethod
    def eis_to_statefinders(theta: np.ndarray) -> List[float]:
        """Convert EIS parameters [H0, E1, E2, ...] to statefinder parameters [H0, q0, j0, s0, ...]."""
        H0 = theta[0]
        E1 = theta[1]
        q0 = E1 - 1
        theta_final = [H0, q0]

        if len(theta) >= 3:
            E2 = theta[2]
            j0 = E1**2 - 2*E1 + E2 + 1
            theta_final.append(j0)

        if len(theta) >= 4:
            E3 = theta[3]
            s0 = -E1**3 + 3*E1**2 - 4*E1*E2 - 3*E1 + E2 - E3 + 1
            theta_final.append(s0)

        if len(theta) >= 5:
            E4 = theta[4]
            l0 = (E1**4 - 4*E1**3 + 11*E1**2*E2 + 6*E1**2 - E1*E2 +
                  7*E1*E3 - 4*E1 + 4*E2**2 + 2*E2 + E3 + E4 + 1)
            theta_final.append(l0)

        if len(theta) >= 6:
            E5 = theta[5]
            p0 = (-E1**5 + 5*E1**4 - 26*E1**3*E2 - 10*E1**3 - 18*E1**2*E2 -
                  32*E1**2*E3 + 10*E1**2 - 34*E1*E2**2 - 18*E1*E2 -
                  24*E1*E3 - 11*E1*E4 - 5*E1 - 11*E2**2 - 15*E2*E3 +
                  2*E2 - 4*E3 - 4*E4 - E5 + 1)
            theta_final.append(p0)

        return theta_final