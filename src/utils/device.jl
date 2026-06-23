### Device utilities
###
### Helper functions for hanlding data between devices (cpu and gpu)



# Define conversion for scaling
function to_cpu(scaling::Scaling)

    return Scaling(
        cpu_device()(scaling.sc_a),
        cpu_device()(scaling.sc_b)
    )
end

# Define conversion for zscore stats
function to_cpu(z::ZScoreStats)

    return ZScoreStats(
        cpu_device()(z.input_mean),
        cpu_device()(z.input_std),
        cpu_device()(z.output_mean),
        cpu_device()(z.output_std),
    )
end



# Define conversion of general LW schemes (everything is already on cpu=
to_cpu(s::AbstractNeuralLW) = s

# Define conversion for global ABRLW scheme
function to_cpu(s::NeuralABRLWGlobal)
    
    return NeuralABRLWGlobal(
        s.n_in, s.n_out, s.n_points,
        s.arch_config,
        to_cpu(s.zscore),
        s.nn, cpu_device()(s.ps), cpu_device()(s.st)
    )
end