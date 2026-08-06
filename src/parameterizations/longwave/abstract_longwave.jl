### Abstract types for longwave parameterizations
#
# Structure:
#
# SpeedyWeather.AbstractLongwave        # longwave radiation parameterization schemes from SW
#   - AbstractLW                        # schemes from this project
#       - ConstLW                       # constant parameters scheme
#       - NeuralLW                      # neural network scheme           



# Common supertype for all longwave schemes in this project
abstract type AbstractLW <: SpeedyWeather.AbstractLongwave end