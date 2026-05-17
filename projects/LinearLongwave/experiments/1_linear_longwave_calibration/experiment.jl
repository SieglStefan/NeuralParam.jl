# Run "include("_setup.jl")" first!

spectral_grid = SpectralGrid(trunc=10, nlayers=1)

L, a, b = run_calibration(spectral_grid, ntime=3, nsteps=10,
                                eta_a = 1f-14, eta_b=1f-10)
plot_calibration(L, a, b)