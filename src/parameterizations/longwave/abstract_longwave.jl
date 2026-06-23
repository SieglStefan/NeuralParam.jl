### Abstract types for longwave parameterizations
#
# Structure:
#
# SpeedyWeather.AbstractLongwave        # longwave radiation parameterization schemes from SW
#   - AbstractNeuralLW                  # schemes from this project
#       - AbstractLinearLW              # schemes using linear output: dT = a * T + b
#           - ConstLinearLW      
#           - NeuralLinearLW
#       - AbstractABRLW                 # schemes emulating AnalyticBandRadiation.jl
#           - NeuralABRLW
#           - NeuralABRLWGlobal



# Common supertype for all longwave schemes in this project
abstract type AbstractNeuralLW <: SpeedyWeather.AbstractLongwave end


# Schemes using linear output: dT = a * T + b 
abstract type AbstractLinearLW <: AbstractNeuralLW end


# Schemes emulating AnalyticBandRadiation.jl
abstract type AbstractABRLW <: AbstractNeuralLW end