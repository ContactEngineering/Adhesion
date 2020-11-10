from Adhesion.Interactions.Electrostatic import ChargePatternsInteraction


epsilon0 = 8.854e-12  # [C/m^2], permittivity in vacuum


def test_sine_wave():
    import numpy as np
    magnitude = 0.01  # [C/m2]
    length = 50e-9  # [m]
    resolution = 1e-9  # [m]
    size = round(length/resolution)
    charge_distribution = np.empty([size, size])
    for index in range(size):
        charge_distribution[index,:] =  magnitude * np.sin(2*np.pi*index/size)
    sine_wave = ChargePatternsInteraction(charge_distribution,
        physical_sizes=(length, length))
    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        colormap = ax.imshow(sine_wave.charge_distribution.T, origin="lower")
        fig.colorbar(colormap)
        plt.show()
    
    gaps = np.linspace(0.1, 10) * length
    stress_numerical = sine_wave.evaluate(gaps, 
        potential=False, gradient=True, curvature=False)[1]
    #                σ^2             d
    # T_ad(d) = -  ─────── exp(- 2π ───)
    #               4 ϵ_0            L
    decay = np.exp(-2 * np.pi * gaps / length)
    stress_analytical = -magnitude**2 / (4*epsilon0) * decay
    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(gaps, stress_numerical, label="numerical")
        ax.plot(gaps, stress_analytical, label="analytical")
        plt.show()
    np.testing.assert_allclose(stress_numerical, stress_analytical, 
        atol=1e-6, rtol=1e-4)
    

def test_sine_wave2D():
    import numpy as np
    magnitude = 0.01  # [C/m2]
    length = 50e-9  # [m]
    resolution = 1e-9  # [m]
    size = round(length/resolution)
    charge_distribution = np.empty([size, size])
    for x, y in np.ndindex(size, size):
        sin_2pi = lambda t: np.sin(2 * np.pi * t)
        charge_distribution[x, y] = sin_2pi(x/size) * sin_2pi(y/size)
    charge_distribution *= magnitude
    sine_wave2D = ChargePatternsInteraction(charge_distribution,
        physical_sizes=(length, length))
    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        colormap = ax.imshow(sine_wave2D.charge_distribution.T, origin="lower")
        fig.colorbar(colormap)
        plt.show()
    
    gaps = np.linspace(0.1, 10) * length
    stress_numerical = sine_wave2D.evaluate(gaps, 
        potential=False, gradient=True, curvature=False)[1]
    #               σ^2               d
    # T_ad(z) = - ─────── exp(-√2 2π ───)
    #              8 ϵ_0              L
    decay = np.exp(-2 * np.pi * gaps / length)
    stress_analytical = -magnitude**2 / (8*epsilon0) * decay**np.sqrt(2)
    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(gaps, stress_numerical, label="numerical")
        ax.plot(gaps, stress_analytical, label="analytical")
        plt.show()
    np.testing.assert_allclose(stress_numerical, stress_analytical, 
        atol=1e-6, rtol=1e-4)