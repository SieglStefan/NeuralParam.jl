using Lux, Random, Optimisers, Enzyme, Reactant

n_in = 8
n_out = 2*n_in
n_layers = 2
width = 32

hidden_layers = [Dense(width => width, tanh) for _ in 1:n_layers]

model = Chain(
    Dense(n_in => width, tanh),
    hidden_layers...,
    Dense(width => n_out))


rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, model)
opt = Optimisers.Adam(1f-4)
opt_state = Optimiser.setup(opt, ps)


train_state = Training.TrainState(mode, ps, st, Adam(0.0001f0))


gs, loss, stats, train_state = Training.single_train_step!(
    AutoEnzyme(),
    MSELoss(),
    (x, dev(rand(rng, Float32, 10, 2))),
    train_state
)