import numpy as np
from Adhesion.Interactions.Electrostatic import ChargePatternsInteraction


epsilon0 = 8.854e-12  # [C/m^2], permittivity in vacuum


def test_sinewave():
    magnitude = 0.01  # [C/m2]
    length = 50e-9  # [m]
    resolution = 1e-9  # [m]
    n = round(length/resolution)  # number of points in one axis
    charge_distribution = magnitude * np.broadcast_to(
        np.sin(2*np.pi*np.arange(n)/n), [n, n])
    sinewave = ChargePatternsInteraction(
        charge_distribution, physical_sizes=(length, length))
    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        colormap = ax.imshow(sinewave.charge_distribution.T, origin="lower")
        fig.colorbar(colormap)
        plt.show()

    gaps = np.linspace(0.1, 10) * length
    stress_numerical = sinewave.evaluate(
        gaps, potential=False, gradient=True, curvature=False
        )[1]
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
    np.testing.assert_allclose(
        stress_numerical, stress_analytical, atol=1e-6, rtol=1e-4
        )


def test_sinewave2D():
    magnitude = 0.01  # [C/m2]
    length = 50e-9  # [m]
    resolution = 1e-9  # [m]
    n = round(length/resolution)  # number of points in one axis
    charge_distribution = magnitude * np.outer(
        np.sin(2*np.pi*np.arange(n)/n), np.sin(2*np.pi*np.arange(n)/n))
    sinewave2D = ChargePatternsInteraction(
        charge_distribution, physical_sizes=(length, length))
    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        colormap = ax.imshow(sinewave2D.charge_distribution.T, origin="lower")
        fig.colorbar(colormap)
        plt.show()

    gaps = np.linspace(0.1, 10) * length
    stress_numerical = sinewave2D.evaluate(
        gaps, potential=False, gradient=True, curvature=False
        )[1]
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
    np.testing.assert_allclose(
        stress_numerical, stress_analytical, atol=1e-6, rtol=1e-4
        )


def test_distribution():
    magnitude = 0.01  # [C/m2]
    length = 50e-9  # [m]
    resolution = 1e-9  # [m]
    n = round(length/resolution)  # number of points in one axis
    charge_distribution = magnitude * np.outer(
        np.sin(2*np.pi*np.arange(n)/n), np.sin(2*np.pi*np.arange(n)/n))
    sinewave2D = ChargePatternsInteraction(
        charge_distribution, physical_sizes=(length, length))
    gaps = np.linspace(0.1, 10, 10) * length
    test_return = sinewave2D.evaluate(
        gaps,
        potential=False,
        stress_dist=True,
        )[3]
    print(test_return)
