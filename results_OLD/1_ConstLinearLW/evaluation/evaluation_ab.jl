### XXX
###
###


#MULTI_RUN = "run_XXX"
#EXCLUDE   = []
#N_SCHEMES = 7

#calib_dir = joinpath(@__DIR__, "..", "calibration", MULTI_RUN)   # <- lokal, wg. @__DIR__
#tasks     = [i for i in 0:N_SCHEMES-1 if !(i in EXCLUDE)]
#SCHEMES_AB = (; (Symbol("task_$i") =>
#    load_scheme(path = joinpath(calib_dir, "task_$i"), file = "scheme.jld2") for i in tasks)...)

#run_evaluation_ab(; schemes = SCHEMES_AB, nlayers = NLAYERS, run_dir = RUN_DIR, folder_name = "initial_ab")