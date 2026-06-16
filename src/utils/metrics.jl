### Metrics and norm utilities
###
### This file contains scalar validation metrics for temperature fields
### and tree norm utilities for parameters/gradients.



# Root mean squared error metric
function rmse(x, y)
    return sqrt(sum((y .- x).^2) / length(x))
end


# Bias metric
function bias(x, y)
    return sum(y .- x) / length(x)
end


# Correlation metric
function correlation(x, y)
    return cor(vec(x), vec(y))
end


# Maximal absolute difference metric
function maxdiff(x, y)
    return maximum(abs.(y .- x))
end



# Recursive squared L2 norm for numbers, arrays, tuples, and NamedTuples
tree_l2sum(x::Number) = abs2(x)
tree_l2sum(x::AbstractArray) = sum(abs2, x)
tree_l2sum(x::Tuple) = sum(tree_l2sum, x)
tree_l2sum(x::NamedTuple) = sum(tree_l2sum, values(x))

# Recursive L2 norm for parameter/gradient trees
tree_l2norm(x) = sqrt(tree_l2sum(x))