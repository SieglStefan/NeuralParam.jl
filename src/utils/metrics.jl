### Metrics utilities
###
### Different metrics and norms for evaluating model performance, training loss and gradients



# Root mean squared error
rmse(x, y) = sqrt(sum(abs2, x .- y) / length(x))
# Bias
bias(x, y) = sum(y .- x) / length(x)

# Weighted metrics
wmean(x, w) = sum(w .* x) / length(x)
wrmse(x, y, w) = sqrt(wmean(abs2.(x .- y), w))
wbias(x, y, w) = wmean(x .- y, w)


# Correlation
correlation(x, y) = cor(vec(x), vec(y))
# Maximal absolute difference
maxdiff(x, y) = maximum(abs.(y .- x))



# Recursive squared L2 norms
tree_l2sum(x::Number)           = abs2(x)
tree_l2sum(x::AbstractArray)    = sum(abs2, x)
tree_l2sum(x::Tuple)            = sum(tree_l2sum, x)
tree_l2sum(x::NamedTuple)       = sum(tree_l2sum, values(x))

# Recursive L2 norm for parameter/gradient trees
tree_l2norm(x) = sqrt(tree_l2sum(x))


# Utility functions for adding parameter/gradient trees
tree_add(a::Number, b::Number)                = a + b
tree_add(a::AbstractArray, b::AbstractArray)  = a .+ b
tree_add(a::Tuple, b::Tuple)                  = map(tree_add, a, b)
tree_add(a::NamedTuple, b::NamedTuple)        = map(tree_add, a, b)

# Utility functions for scaling parameter/gradient trees
tree_scale(a::Number, s)        = a * s
tree_scale(a::AbstractArray, s) = a .* s
tree_scale(a::Tuple, s)         = map(x -> tree_scale(x, s), a)
tree_scale(a::NamedTuple, s)    = map(x -> tree_scale(x, s), a)