using Lux, Random, Optimisers, Enzyme, Reactant

# Seeding
rng = Random.default_rng()
Random.seed!(rng, 0)

# Construct the layer
model = Chain(Dense(128, 256, tanh), Chain(Dense(256, 256, tanh), Dense(256, 10)))



# Get the device determined by Lux
dev = reactant_device()

# Parameter and State Variables
ps, st = Lux.setup(rng, model) |> dev

# Dummy Input
x = rand(rng, Float32, 128, 2) |> dev

# Run the model
## We need to use @jit to compile and run the model with Reactant
y, st = @jit Lux.apply(model, x, ps, st)

## For best performance, first compile the model with Reactant and then run it
apply_compiled = @compile Lux.apply(model, x, ps, st)
apply_compiled(model, x, ps, st)

# Gradients
## First construct a TrainState
train_state = Training.TrainState(model, ps, st, Adam(0.0001f0))

## We can compute the gradients using Training.compute_gradients
## TrainState handles compilation internally
gs, loss, stats, train_state = Lux.Training.compute_gradients(
    AutoEnzyme(),
    MSELoss(),
    (x, dev(rand(rng, Float32, 10, 2))),
    train_state
)

## Optimization
train_state = Training.apply_gradients!(train_state, gs) # or Training.apply_gradients (no `!` at the end)

# Both these steps can be combined into a single call (preferred approach)
gs, loss, stats, train_state = Training.single_train_step!(
    AutoEnzyme(),
    MSELoss(),
    (x, dev(rand(rng, Float32, 10, 2))),
    train_state
)