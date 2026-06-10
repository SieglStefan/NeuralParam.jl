### Test file for crashing Julia REPL
###
### Important: First run setup.jl: experiments/setup.jl (one layer abouve this .jl)
### Uses Lux.apply() in SpeedyWeather.parameterization!() (parameterizations/neural_llw.jl)
### Crahes REPL after about ~ 1 hour
### Crash message:  The terminal process "julia.exe '-i', '--banner=no', '--project=c:\Code\sw-ml-thesis', 
#                   'c:\Users\stefa\.vscode\extensions\julialang.language-julia-1.215.2\scripts\terminalserver\terminalserver.jl', 
#                   '\\.\pipe\vsc-jl-repl-a8d0d154-e423-465a-80fe-f42405a298ee', '\\.\pipe\vsc-jl-repldbg-89ee4056-3d6f-45e8-9169-dfb1295a034a', 
#                   '\\.\pipe\vsc-jl-cr-32767007-8faf-4e64-940a-0bc9d04d3eaf', 'USE_REVISE=true', 'USE_PLOTPANE=true', 'USE_PROGRESS=true', 
#                   'ENABLE_SHELL_INTEGRATION=true', 'DEBUG_MODE=false', 'PLOTS_DEFAULT_MIME=image/png'" terminated with exit code: -1073741819.




using SpeedyWeather

spectral_grid = SpectralGrid(trunc=31, nlayers=8)

config = NeuralLinearLongwaveConfig(
    spectral_grid;
    width = 8,
    n_hidden = 1,
)

radiation = NeuralLinearLongwave(config)


L, P, G, PN, GN = run_training!(radiation, 
                        spectral_grid;
                        eta0 = 1f-3,

                        n_ic = 3,              
                        n_traj = 10,
                        n_epochs = 10,
                        n_gap = 30,
                        n_steps = 10,

                        printing_ic = true,
                        printing_traj = true,
                        printing_epochs = true,

                        test_mode = false)


plot_training(L, P, G, PN, GN)