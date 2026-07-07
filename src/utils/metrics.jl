### Metrics and norm utilities
###
### Helper functions for evaluation data



# Mean squared error
function mse(x, y)
    return sum(abs2, (y .- x)) / length(x)
end

# Root mean squared error
function rmse(x, y)
    return sqrt(mse(x, y))
end

# Normalized mean squared error
function norm_mse(x, y, std)
    return sum(abs2, (x .- y) ./ std) / length(x)
end


# Bias
function bias(x, y)
    return sum(y .- x) / length(x)
end

# Correlation
function correlation(x, y)
    return cor(vec(x), vec(y))
end

# Maximal absolute difference
function maxdiff(x, y)
    return maximum(abs.(y .- x))
end



# Recursive squared L2 norms
tree_l2sum(x::Number) = abs2(x)
tree_l2sum(x::AbstractArray) = sum(abs2, x)
tree_l2sum(x::Tuple) = sum(tree_l2sum, x)
tree_l2sum(x::NamedTuple) = sum(tree_l2sum, values(x))

# Recursive L2 norm for parameter/gradient trees
tree_l2norm(x) = sqrt(tree_l2sum(x))