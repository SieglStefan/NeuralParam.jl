using CUDA
using Reactant

if CUDA.functional()
    Reactant.set_default_backend("gpu")
    print("GPU backend set as default.")
else
    Reactant.set_default_backend("cpu")
    print("CPU backend set as default.")
end