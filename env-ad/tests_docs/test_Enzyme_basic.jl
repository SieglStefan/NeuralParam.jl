using Enzyme

function f(x::Array{Float64}, y::Array{Float64})
    y[1] = x[1] * x[1] + x[2] * x[1]
    return nothing
end;

x  = [2.0, 2.0]
bx = [0.0, 0.0]
y  = [0.0]
by = [1.0];

Enzyme.autodiff(Reverse, f, Duplicated(x, bx), Duplicated(y, by));

@show bx