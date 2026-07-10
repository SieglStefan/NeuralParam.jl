### Init-Unabhängigkeits-Test: kalibrierte a,b über verschiedene Startwerte vergleichen
###
### Lädt die Schemata aus calibration/init_test/job_XXXX/init_N/scheme.jld2,
### printet alle a,b der Reihe nach und plottet sie (eine Linie pro init).
### Decken sich alle Linien -> eindeutiges Minimum, start-unabhängig.

using NeuralParam
using Plots


# --- Schemata finden & laden (Job-Ordner automatisch) ---------------------
init_test = joinpath(@__DIR__, "..", "calibration", "init_test")

job  = last(sort(filter(d -> startswith(d, "job_"),  readdir(init_test))))   # neuester Job
base = joinpath(init_test, job)

EXCLUDE = ["init_4", "init_3"]                                                          # welche weglassen
init_names = sort(filter(d -> startswith(d, "init_") && d ∉ EXCLUDE, readdir(base)))   # init_0, init_1, ...
schemes    = [load_scheme(path = joinpath(base, d), file = "scheme.jld2") for d in init_names]

NLAYERS = length(schemes[1].ps.a)
layers  = 1:NLAYERS

println("Job: $job   ($(length(schemes)) Schemata)\n")


# --- Alle a,b der Reihe nach printen --------------------------------------
for (name, s) in zip(init_names, schemes)
    println(name, ":")
    println("  a = ", s.ps.a)
    println("  b = ", s.ps.b)
    println()
end


# --- Plotten: a und b pro Layer, eine Linie pro init ----------------------
p_a = plot(; xlabel = "a", ylabel = "Layer", title = "Calibrated a",
           yflip = true, yticks = layers, legend = :bottomleft)
for (name, s) in zip(init_names, schemes)
    Plots.plot!(p_a, s.ps.a, layers; label = name, marker = :circle, lw = 2)
end

p_b = plot(; xlabel = "b", ylabel = "Layer", title = "Calibrated b",
           yflip = true, yticks = layers, legend = :bottomleft)
for (name, s) in zip(init_names, schemes)
    Plots.plot!(p_b, s.ps.b, layers; label = name, marker = :circle, lw = 2)
end

fig = plot(p_a, p_b; layout = (1, 2), plot_title = "Init-independence: calibrated a,b")
display(fig)


# --- Speichern ------------------------------------------------------------
out_dir = joinpath(@__DIR__, "multi_ab")
mkpath(out_dir)
Plots.savefig(fig, joinpath(out_dir, "multi_ab.png"))
