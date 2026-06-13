

using Random

rng=Random.default_rng()




noise = randn!(rng, similar(ones(1000000000,1)))

maximum(noise)