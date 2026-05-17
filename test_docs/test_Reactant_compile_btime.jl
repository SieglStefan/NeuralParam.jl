using Reactant, BenchmarkTools

input1 = Reactant.ConcreteRArray(ones(10))
input2 = Reactant.ConcreteRArray(ones(10))

function sinsum_add(x,y)
    return sum(sin.(x) .+ y)
end

f = @compile sinsum_add(input1, input2)

@btime $f(input1, input2)
@btime $sinsum_add(input1, input2)