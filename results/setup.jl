### Convenience script for loading NeuralParam for experiments

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))          

include(joinpath(@__DIR__, "..", "src", "NeuralParam.jl"))

using .NeuralParam                                        