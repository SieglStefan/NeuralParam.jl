### Statistics helpers
###
### Used for building the stats file (see zscore.jl)



# Mean and std over all entries
mean_std(x) = (f = filter(isfinite, x); (; mean = Float32[mean(f)], std = Float32[std(f)]))

# Mean and std per vertical layer (averaging over all other dimensions)
mean_std_layers(x) = (;
    mean = Float32[mean(filter(isfinite, selectdim(x, 2, k))) for k in axes(x, 2)],
    std  = Float32[std(filter(isfinite, selectdim(x, 2, k)))  for k in axes(x, 2)],
)


# Fit y = a + b*x for one column
function fit_linear(x::AbstractVector, y::AbstractVector)
    x_bar = mean(x)
    y_bar = mean(y)

    Sxx = sum(abs2, x .- x_bar)
    b = Sxx > 0 ? sum((x .- x_bar) .* (y .- y_bar)) / Sxx : 0f0

    return Float32(y_bar - b*x_bar), Float32(b)     # intercept, slope
end

# Fit y = a + b*x1 + c*x2 for one column
function fit_linear(x1::AbstractVector, x2::AbstractVector, y::AbstractVector)
    A = hcat(ones(Float32, length(y)), Float32.(x1), Float32.(x2))
    c = A \ Float32.(y)
    return c[1], c[2], c[3]                         # a, b, c
end