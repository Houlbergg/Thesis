# Using the formula v_e = flow_factor * ( Q / A_e )
# A_e = w_e * h_e * epsilon
# where epsilon is the porosity of the electrode.
# This effectively reduces the area of the electrode face by "removing" the area not available to the flow.
electrodes = ["Felt", "Cloth", "Paper"]  # Electrodes
Q_Baichen = [
    67.5,
    34.4,
    16.9,
]  # Volumetric flow rate from Baichen in mL/min.
# For 0% Compression, Felt, Cloth and Paper respectively.
# He uses a simple v_e = Q / A_e. The flow rates given correspond to v_e = 2 cm/s
# He states that felt works fine at 20-40%, but cloth is not good > 20%. Paper should be fine in 0-30%
v_e = [0.5, 1, 2, 5]  # Flow velocities from Baichen in cm/s
w_e = [2.3]  # Width of electrode profile in cm
h_e = [
    0.25,
    0.0406,
    0.0201,
]  # Height of electrode profiles in cm. 0% compression. Felt, Cloth, Paper.
epsilon = [0.93]  # Porosity of the electrode.
flow_factor = [7, 8, 10]
# Referenced value based on the following article.
# Should be recalculated for each electrode
# https://www-sciencedirect-com.proxy.findit.cvt.dk/science/article/pii/S037877531301567X
# Flow factor is somewhat arbitrary. But based on the two following articles should be somewhere between 7 and 10
# https://www-sciencedirect-com.proxy.findit.cvt.dk/science/article/pii/S037877531301567X
# https://www.sciencedirect.com/science/article/pii/S2405896322004189
CRs = [0.3, 0.2, 0.2]  # Compression ratios for Felt, Cloth, Paper
# Calculate the new flow rate.
# Simple case Q = v_e * A_e
# Expanded case:
# Q = ( v_e * A_e ) / flow_factor
# Q = v_e * (w_e * h_e * epsilon) / flow_factor

# Unit corrections. w_e, h_e, v_e / 100 to go from cm to m
# Q * 1 000 000 to go from m^3/s to mL/s
# Q * 60 to go from mL/s to mL/min
# Q * 60 000 to go from L/s to mL/min
Q_CR0 = {}
Q_CRs = {}

for i, elec in enumerate(electrodes):
    Q0s = []
    Qs = []
    for j, v in enumerate(v_e):
        Q0s.append(60 * v * w_e[0] * h_e[i])
        Qs.append(60 * v * w_e[0] * h_e[i] * (1 - CRs[i]))
    Q_CR0[elec] = Q0s
    Q_CRs[elec] = Qs
Q_expanded = {}

for flow in flow_factor:
    Q_s = []
    for v in v_e:
        Q_s.append((60 * v * w_e[0] * h_e[0] * epsilon[0]) / flow)
    Q_expanded[flow] = Q_s

A_geo = 2.3**2  # Area of the electrode face in cm^2
current_density = 75  # mA/cm^2
current = current_density * A_geo  # Current in mA
n = 1  # Number of electrons
F = 96485  # Faraday constant


def k_m(v):  #
    return 1.6 * 10 ** (-4) * v**0.4


C_b = 1  # Molar concentration of the electrolyte in mol/L
Q_old = [5, 10, 20, 50]  # Volumetric flow rate in mL/min
i_L = []

for vv in v_e:
    i_L.append(n * F * k_m(vv))

Q_naive = ((current / 1000) / (n * F * C_b)) * 60000  # Naive flow rate in mL/min