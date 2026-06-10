### Abstract types for NeuralLongwave parameterizations
#
# Structure:
#
# SpeedyWeather.AbstractLongwave        # LW radiation parameterization schemes from SW
#   - AbstractNeuralLongwave            # schemes from this project
#       - AbstractLuxLongwave           # schemes that use a Lux NN
#           - NeuralLinearLongwave      
#           - NeuralABRLongwave
#       - AbstractConstLongwave         # schemes that use const. parameters
#           - ConstLinearLongwave



# Common supertype for all longwave schemes in this project
abstract type AbstractNeuralLongwave <: SpeedyWeather.AbstractLongwave end


# Schemes that use a Lux NN
abstract type AbstractLuxLongwave <: AbstractNeuralLongwave end

# Schemes that use const. parameters
abstract type AbstractConstLongwave <: AbstractNeuralLongwave end