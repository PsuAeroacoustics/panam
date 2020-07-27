import numpy as np


def gopalan_thickness(M_AT, M_H, A, r):
    o_elv = 0.0
    o_psi = 0.0
    psis = np.linspace(0, 2.0 * np.pi, 1000)
    mu = M_AT / M_H - 1
    mue = mu * np.cos(o_psi)
    MHe = M_H / (1 - mue * M_H)
    MHe = MHe * np.cos(o_elv)

    C = np.cos(psis)
    S = np.sin(psis)

    f11 = -MHe ** 3 / 24 * S * (3 - MHe * S)
    f12 = -MHe ** 3 / 24 * (-3 + (3 + 2 * S ** 2) * (3 - MHe * S * (3 - MHe * S)))
    term1 = (f11 + mue * f12) / (1 - MHe * S) ** 3

    f21 = MHe ** 4 / 240 * (
            50 - 45 * MHe * S + 39 * MHe ** 2 - 11 * MHe ** 2 * S ** 2 + 12 * MHe ** 3 * S - 18 * MHe ** 3 * S ** 3)
    f22 = MHe ** 3 / 240 * (
            180 - 480 * MHe * S + 120 * MHe ** 2 + 520 * MHe ** 2 * S ** 2 - 195 * MHe ** 3 * S - 280 * MHe ** 3
            * S ** 3 + 138 * MHe ** 4 * S ** 2 + 60 * MHe ** 4 * S ** 4 - 36 * MHe ** 5 * S ** 3)
    term2 = C ** 2 * (f21 + mue * f22) / (1 - MHe * S) ** 4

    f32 = MHe ** 3 / 240 * (-36 * MHe ** 5 * S ** 2 * C ** 2 + 290 * MHe ** 3 - 570 * MHe - 60 * MHe ** 4 * S
                            * (1 + C ** 4) + 300 * MHe ** 3 * C ** 4 + 198 * MHe ** 4 * C ** 2 * S
                            - 560 * MHe ** 2 * S - 240 * S - 600 * MHe * C ** 2
                            * (1 - MHe * S) - 605 * MHe ** 3 * C ** 2)
    term3 = C * (np.sin(o_psi) * mu * f32) / (1 - MHe * S) ** 4

    C_p_t = A / r * (term1 + term2 + 0 * term3) / (np.cos(o_elv) ** 2)

    a0 = 343
    rho0 = 1.225
    return psis, C_p_t * rho0 * a0 ** 2
