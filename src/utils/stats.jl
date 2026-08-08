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
    u = Float64.(x)
    v = Float64.(y)

    x_bar = mean(u)
    y_bar = mean(v)

    dx  = u .- x_bar
    Sxx = sum(abs2, dx)
    b   = Sxx > 0 ? sum(dx .* (v .- y_bar)) / Sxx : 0.0

    return Float32(y_bar - b*x_bar), Float32(b)     # intercept, slope
end

# Fit y = a + b*x1 + c*x2 for one column
function fit_linear(x1::AbstractVector, x2::AbstractVector, y::AbstractVector)
    u1 = Float64.(x1);  s1 = std(u1);  s1 = s1 > 0 ? s1 : 1.0
    u2 = Float64.(x2);  s2 = std(u2);  s2 = s2 > 0 ? s2 : 1.0

    A = hcat(ones(length(y)), u1 ./ s1, u2 ./ s2)
    c = A \ Float64.(y)

    return Float32(c[1]), Float32(c[2] / s1), Float32(c[3] / s2)
end