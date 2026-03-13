"""
This code should plot the models for 
NACA 2312 and 2324 airfoils,
and the NACA 4412 and 4424 airfoils
including camber line

"""

import numpy as np # trying to import numpy functions
import matplotlib.pyplot as plt # I think I will try to use this to plot


def shape_naca(m, p, xx, c=1, N=200):
    """
    This function takes inputs as the NACA airfoil codes (2312, 4412, etc.)
    and plots the airfoil surface. I think it will break if camber does not equal 1.  
    """
    m = m/100           # 1stchar in NACA code, max camber as a percentage of the chord
    p = p/10            # 2nd char in NACA code, location of the max camber as a percentage of the chord
    xx = xx/100         # 3rd and 4th char in NACA code, thickness from the last 2 characters as a percentage of chord length
    t = c*xx            # max thickness
    x_vals = np.linspace(0, c, N)   # linspace is better than arange since we need c inclusive

    pos_camber = np.zeros(shape=(N,2))          # arrays for storing data to plot
    pos_upper = np.zeros(shape=(N,2))    
    pos_lower = np.zeros(shape=(N,2))    

    for i, x in enumerate(x_vals):            # enumerate to track iteration counter
        
        xi = x/c                      # xi is the position between 0 and c along chord
        if xi >= 0 and xi <= p:                     # camber line based on class equations
            y_camber = (m/p**2)*(2*p*xi - xi**2)    # this include derivatives I calulated so could be wrong
            dy_dx = 2*m/(c*p**2)*(p - xi)
        elif xi > p and xi <= 1:
            y_camber = (m/(1-p)**2)*(1 - 2*p + 2*p*xi - xi**2)
            dy_dx = 2*m/(c*(1-p)**2)*(p - xi)
        else:
            break
        
        pos_camber[i] = [x, y_camber]       # storing camber data    
        
        y_thick = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*(x)**2 + 0.2843*(x)**3 - 0.1015*(x)**4)        # Equation for thickness
        
        theta = np.arctan(dy_dx)                  # upper and lower surface positions calculated 
        x_u = x - y_thick*np.sin(theta)         # using theta
        y_u = y_camber + y_thick*np.cos(theta)
        x_l = x + y_thick*np.sin(theta)
        y_l = y_camber - y_thick*np.cos(theta)

        pos_upper[i] = [x_u, y_u]               # upper surface position array
        pos_lower[i] = [x_l, y_l]               # lower

    return pos_camber, pos_upper, pos_lower


"""
# NACA 2312

camber, upper, lower = shape_naca(2, 3, 12)             # NACA code is the input, not the percentages/decimals
plt.close('all')                                    
fig, ax = plt.subplots()
ax.plot(upper[:, 0], upper[:, 1], 'r', label="$NACA 2312$")
ax.plot(lower[:, 0], lower[:, 1], 'r')
ax.plot(camber[:,0], camber[:,1], 'k')

# NACA 2324

camber, upper, lower = shape_naca(2, 3, 24)
ax.plot(upper[:, 0], upper[:, 1], 'b', label="$NACA 2324$")
ax.plot(lower[:, 0], lower[:, 1], 'b')
ax.plot(camber[:,0], camber[:,1], 'k')
ax.axis('equal')
ax.set_title('Compare NACA 2312 and 2324 airfoils')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc='best')
plt.grid()
plt.show()

# NACA 4412

plt.close('all') 
camber, upper, lower = shape_naca(4, 4, 12)
fig, ax = plt.subplots()
ax.plot(upper[:, 0], upper[:, 1], 'r', label=f"$NACA 4412$")
ax.plot(lower[:, 0], lower[:, 1], 'r')
ax.plot(camber[:,0], camber[:,1], 'k')

# NACA 4424

camber, upper, lower = shape_naca(4, 4, 24)
ax.plot(upper[:, 0], upper[:, 1], 'b', label=f"$NACA 4424$")
ax.plot(lower[:, 0], lower[:, 1], 'b')
ax.plot(camber[:,0], camber[:,1], 'k')
ax.axis('equal')
ax.set_title('Compare NACA 4412 and 4424 airfoils')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc='best')
plt.grid()
plt.show()

"""

# ============================================================
# THIN AIRFOIL THEORY - Exercises 2 and 3
# ============================================================

def dyc_dx_func(x, m, p, c=1):
    """
    Analytical derivative of the camber line dy_c/dx
    x     : position along chord (physical units)
    m, p  : NACA parameters (already as fractions, e.g. 0.02, 0.3)
    """
    xi = x / c  # normalize
    if xi <= p:
        return (2 * m / (p**2)) * (p - xi) / c
    else:
        return (2 * m / ((1 - p)**2)) * (p - xi) / c


def thin_airfoil(m_raw, p_raw, alpha_deg, c=1, N=1000):
    """
    Computes Cl and delta_Cp using thin airfoil theory.
    
    Inputs:
        m_raw     : first NACA digit (e.g. 2 for NACA 2312)
        p_raw     : second NACA digit (e.g. 3 for NACA 2312)
        alpha_deg : angle of attack in degrees
        c         : chord length (default 1)
        N         : number of integration points
    
    Returns:
        Cl        : lift coefficient (scalar)
        x_vals    : x/c positions for delta_Cp (array)
        delta_Cp  : pressure difference distribution (array)
    """
    m = m_raw / 100   # max camber as fraction
    p = p_raw / 10    # location of max camber as fraction

    alpha_rad = np.deg2rad(alpha_deg)

    # --- Theta array (avoid exact 0 and pi to prevent sin=0 issues) ---
    #theta = np.linspace(1e-6, np.pi - 1e-6, N)
    theta = np.linspace(1e-1, np.pi - 1e-6, N)

    # --- x/c positions corresponding to theta ---
    x = 0.5 * c * (1 - np.cos(theta))  # physical x values

    # --- Camber slope at each x ---
    slope = np.array([dyc_dx_func(xi, m, p, c) for xi in x])

    # --- Fourier coefficients ---
    A0 = alpha_rad - (1 / np.pi) * np.trapezoid(slope, theta)
    A1 = (2 / np.pi) * np.trapezoid(slope * np.cos(theta), theta)

    # --- Lift coefficient ---
    Cl = 2 * np.pi * (A0 + A1 / 2)

    # --- Pressure difference distribution ---
    # gamma/U_inf = 2 * (A0*(1+cos)/sin + A1*sin + ...)
    # we keep only A0 and A1 terms (higher terms are negligible)
    delta_Cp = 2 * (A0 * (1 + np.cos(theta)) / np.sin(theta)
                    + A1 * np.sin(theta))

    x_normalized = x / c  # return x/c not physical x

    results = {
        "Cl": Cl,
        "x/c": x_normalized,
        "delta_Cp": delta_Cp
    }
    return results


# ============================================================
# EXERCISE 2: Cl vs AoA for all 4 airfoils
# ============================================================
'''
airfoils = {
    'NACA 2312': (2, 3, 12),
    'NACA 2324': (2, 3, 24),
    'NACA 4412': (4, 4, 12),
    'NACA 4424': (4, 4, 24),
}

alphas = np.linspace(-10, 15, 50)  # degrees

plt.close('all')

for name, (m_raw, p_raw, xx) in airfoils.items():
    Cl_list = []
    for alpha in alphas:
        Cl, _, _ = thin_airfoil(m_raw, p_raw, alpha)
        Cl_list.append(Cl)

    fig, ax = plt.subplots()
    ax.plot(alphas, Cl_list, 'k', label='Thin Airfoil Theory')
    ax.set_xlabel('Angle of Attack (degrees)')
    ax.set_ylabel('$C_l$')
    ax.set_title(f'Lift Coefficient vs AoA — {name}')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()


# ============================================================
# EXERCISE 3: Delta_Cp vs x/c at AoA = 10 degrees
# ============================================================

alpha_fixed = 10  # degrees

plt.close('all')

for name, (m_raw, p_raw, xx) in airfoils.items():
    _, x_norm, delta_Cp = thin_airfoil(m_raw, p_raw, alpha_fixed)

    fig, ax = plt.subplots()
    ax.plot(x_norm, delta_Cp, 'k', label='Thin Airfoil Theory')
    ax.set_xlabel('x/c')
    ax.set_ylabel('$\\Delta C_p$')
    ax.set_title(f'Pressure Difference Distribution — {name}, AoA = {alpha_fixed}°')
    ax.legend()
    ax.grid()
    #ax.set_yscale('symlog')
    plt.tight_layout()
    plt.show()
'''