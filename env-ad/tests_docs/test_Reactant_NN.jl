# from Reactant.jl documentation

using Reactant, Enzyme, Lux

# Define network
function neural_net(x, w1, w2, b1, b2)
    h = tanh.(w1 * x .+ b1)
    return w2 * h .+ b2
end

# Loss function
function loss(x, y, w1, w2, b1, b2)
    pred = neural_net(x, w1, w2, b1, b2)
    return sum(abs2, pred .- y)
end

# Generate data
x = Reactant.to_rarray(rand(Float32, 10, 32))
y = Reactant.to_rarray(2 .* sum(abs2, Array(x); dims=1) .+ rand(Float32, 1, 32) .* 0.001f0)

# Initialize parameters
w1 = Reactant.to_rarray(rand(Float32, 20, 10))
w2 = Reactant.to_rarray(rand(Float32, 1, 20))
b1 = Reactant.to_rarray(rand(Float32, 20))
b2 = Reactant.to_rarray(rand(Float32, 1))

# Training step
function train_step(x, y, w1, w2, b1, b2, lr)
    # Compute gradients
    (; val, derivs) = Enzyme.gradient(
        ReverseWithPrimal, loss, Const(x), Const(y), w1, w2, b1, b2
    )

    # Update parameters (simple gradient descent)
    w1 .-= lr .* derivs[3]
    w2 .-= lr .* derivs[4]
    b1 .-= lr .* derivs[5]
    b2 .-= lr .* derivs[6]

    return val, w1, w2, b1, b2
end

# Training loop
compiled_train_step = @compile train_step(x, y, w1, w2, b1, b2, 0.001f0)

for epoch in 1:100
    global w1, w2, b1, b2
    loss_val, w1, w2, b1, b2 = compiled_train_step(x, y, w1, w2, b1, b2, 0.001f0)
    if epoch % 10 == 0
        @info "Epoch: $epoch, Loss: $loss_val"
    end
end

println("Training completed!")