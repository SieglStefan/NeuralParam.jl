using Pkg

cd("env-dev")
Pkg.activate(".")

using SpeedyWeather
print(pathof(SpeedyWeather))