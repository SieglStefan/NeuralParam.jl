### Functions for plotting zscore statistics
###
### - Histograms of the z-score normalized inputs and outputs (should look ~N(0,1))
### - Regression diagnostics for the linear output form:
###       dT   = a*T + b              one panel per layer
###       olw  = c + d*T[mid_layer] + e*Ts    one panel per predictor
###       slwd = f + g*T[nlayers]     one panel
###
### The coefficients are fitted per column, the scatter pools all columns and the red
### line uses the column-MEAN coefficients — i.e. exactly what ConstLW starts from.
### The spread around the line is therefore the error a global linear scheme makes.



### Histograms

# Generic: one histogram panel per entry of data
function plot_histograms(data, titles; ncols=4, nbins=50, logx=false, suptitle="", size=(1500, 650))
    fig = CairoMakie.Figure(; size)
    for (k, (d, title)) in enumerate(zip(data, titles))
        d = filter(isfinite, d)
        logx && (d = log10.(filter(>(0), d)))
        ax = CairoMakie.Axis(fig[cld(k, ncols), mod1(k, ncols)];
                             title, titlesize = 12,
                             xlabel = logx ? "log₁₀" : "",
                             xticks = CairoMakie.LinearTicks(4),
                             xticklabelrotation = π/5, xticklabelsize = 10)
        CairoMakie.hist!(ax, d; bins = nbins)
        CairoMakie.text!(ax, 0.03, 0.97;
                         text = "μ=$(round(mean(d), sigdigits=4))\nσ=$(round(std(d), sigdigits=4))",
                         space = :relative, align = (:left, :top), fontsize = 9)
    end
    isempty(suptitle) || CairoMakie.Label(fig[0, :], suptitle; fontsize = 17, font = :bold)
    return fig
end


# Histograms of one normalized PROFILE variable, one panel per vertical layer
function plot_profile_histograms(x, st; name = "", ncols = 4)

    nlayers = Base.size(x, 2)

    # Normalize each layer with its own mean/std
    data   = [vec(zscore(selectdim(x, 2, k), st.mean[k], st.std[k])) for k in 1:nlayers]
    titles = ["Layer $k" for k in 1:nlayers]

    return plot_histograms(data, titles; ncols,
        suptitle = "Normalized $name — per layer",
        size     = (350*ncols, 300*cld(nlayers, ncols)))
end


# Histograms of several normalized SCALAR variables, all in one figure
function plot_scalar_histograms(samples, st, names; suptitle = "Normalized scalars", ncols = 3)

    # Look both the samples and their stats up by name
    data   = [vec(zscore(getproperty(samples, n),
                         getproperty(st, n).mean[1],
                         getproperty(st, n).std[1])) for n in names]
    titles = [string(n) for n in names]

    return plot_histograms(data, titles; ncols, suptitle,
        size = (350*ncols, 300*cld(length(names), ncols)))
end



### Regression diagnostics

# Thin several equally long vectors down to at most max_pts points (shared selection)
function thin(max_pts, vecs...)
    n = length(first(vecs))
    n <= max_pts && return vecs
    sel = randperm(n)[1:max_pts]
    return map(v -> v[sel], vecs)
end


# dT = a*P(T) + b, fitted per column and layer — one panel per layer
function plot_regression_dT(samples, lin, form = LinearOutput(); ncols = 4, max_pts = 3000)

    nlayers = Base.size(samples.T, 2)
    fig = CairoMakie.Figure(size = (350*ncols, 300*cld(nlayers, ncols)))

    # Plot against the predictor the coefficients were fitted on
    P = predictor(form, samples.T)

    for k in 1:nlayers
        r, c = fldmod1(k, ncols)
        ax = CairoMakie.Axis(fig[r, c]; xlabel = "P(T)", ylabel = "dT (K/s)", title = "Layer $k")

        x, y = thin(max_pts, vec(P[:,k,:]), vec(samples.dT[:,k,:]))
        CairoMakie.scatter!(ax, x, y; markersize = 2, alpha = 0.2)

        # Line uses the column-MEAN coefficients, the scatter pools all columns,
        # so a spread around the line is expected — it is not a bad fit
        xr = range(extrema(x)...; length = 2)
        CairoMakie.lines!(ax, xr, lin.a.mean[k] .* xr .+ lin.b.mean[k]; color = :red)
    end

    CairoMakie.Label(fig[0, :],
        "$(nameof(typeof(form))): dT = a*P(T) + b   (red: column-mean a, b)"; fontsize = 17, font = :bold)
    return fig
end


# Both flux regressions in one figure:
#   olw  = c + d*P(T[mid_layer]) + e*P(Ts)  two predictors, one panel each with the other held at its mean
#   slwd = f + g*P(T[nlayers])              one predictor, plain scatter with its line
function plot_regression_flux(samples, lin, form = LinearOutput(); max_pts = 5000)

    nlayers = Base.size(samples.T, 2)

    # Plot against the predictors the coefficients were fitted on
    #   - transform once: predictor() copies the whole profile array
    P  = predictor(form, samples.T)
    T1 = P[:, mid_layer(nlayers), :]                         # olw layer temperature
    Tb = P[:, nlayers, :]                                    # bottom-layer temperature
    Ts = predictor(form, samples.Ts)                         # surface temperature

    # Column-mean coefficients — what ConstLW starts from
    c, d, e = lin.c.mean[1], lin.d.mean[1], lin.e.mean[1]
    f, g    = lin.f.mean[1], lin.g.mean[1]

    fig = CairoMakie.Figure(size = (1050, 340))

    # (1) olw vs T[mid_layer], with Ts held at its mean so a line can be drawn
    ax1 = CairoMakie.Axis(fig[1,1]; xlabel = "P(T[$(mid_layer(nlayers))])", ylabel = "olw (W/m²)",
                          title = "olw vs mid-layer T")
    x, y = thin(max_pts, vec(T1), vec(samples.olw))
    CairoMakie.scatter!(ax1, x, y; markersize = 2, alpha = 0.2)
    xr = range(extrema(x)...; length = 2)
    CairoMakie.lines!(ax1, xr, c .+ d .* xr .+ e * mean(Ts); color = :red)

    # (2) olw vs Ts, with T[mid_layer] held at its mean
    ax2 = CairoMakie.Axis(fig[1,2]; xlabel = "P(Ts)", ylabel = "olw (W/m²)",
                          title = "olw vs surface T")
    x, y = thin(max_pts, vec(Ts), vec(samples.olw))
    CairoMakie.scatter!(ax2, x, y; markersize = 2, alpha = 0.2)
    xr = range(extrema(x)...; length = 2)
    CairoMakie.lines!(ax2, xr, c .+ d * mean(T1) .+ e .* xr; color = :red)

    # (3) slwd vs T[nlayers], only one predictor
    ax3 = CairoMakie.Axis(fig[1,3]; xlabel = "P(T[nlayers])", ylabel = "slwd (W/m²)",
                          title = "slwd vs bottom-layer T")
    x, y = thin(max_pts, vec(Tb), vec(samples.slwd))
    CairoMakie.scatter!(ax3, x, y; markersize = 2, alpha = 0.2)
    xr = range(extrema(x)...; length = 2)
    CairoMakie.lines!(ax3, xr, f .+ g .* xr; color = :red)

    CairoMakie.Label(fig[0, :],
        "$(nameof(typeof(form))): olw = c + d*P(T[$(mid_layer(nlayers))]) + e*P(Ts)        " *
        "slwd = f + g*P(T[nlayers])        (red: column-mean coefficients)";
        fontsize = 15, font = :bold)
    return fig
end



### Collection

# Create all validation plots of a zscore statistics run and store them as .png
#   - only the groups actually present in stats are plotted, so dropping an output form
#     from the generation does not break the plotting
function plot_zscore(samples, stats; dir, output_forms = (DirectOutput(), LinearOutput(), PlanckOutput()))

    # Create output directory
    mkpath(dir)

    # One figure per entry, the key becomes the file name
    plots = (;
        # Normalized inputs: one .png per profile variable, all scalars in one
        input_T       = plot_profile_histograms(samples.T, stats.inputs.T; name = "T"),
        input_q       = plot_profile_histograms(samples.q, stats.inputs.q; name = "log10(q)"),
        input_scalar  = plot_scalar_histograms(samples, stats.inputs, (:p, :lat, :lf, :sst, :lst, :Ts);
                                               suptitle = "Normalized scalar inputs"),
    )

    # Normalized direct outputs: dT per layer, fluxes together
    if haskey(stats, :direct)
        plots = merge(plots, (;
            output_dT     = plot_profile_histograms(samples.dT, stats.direct.dT; name = "dT"),
            output_scalar = plot_scalar_histograms(samples, stats.direct, (:olw, :slwd);
                                                   suptitle = "Normalized scalar outputs"),
        ))
    end

    # Regression diagnostics, one pair of figures per fitted affine output form
    for form in output_forms

        form isa DirectOutput && continue           # has no regression, plotted above
        group = output_group(form)
        haskey(stats, group) || continue

        plots = merge(plots, (;
            (Symbol("regression_$(group)_dT")   => plot_regression_dT(samples, stats[group], form),
             Symbol("regression_$(group)_flux") => plot_regression_flux(samples, stats[group], form))...,
        ))
    end

    # Save plots
    for (plot_name, fig) in pairs(plots)
        CairoMakie.save(joinpath(dir, "$(plot_name).png"), fig)
    end

    @info "Plots stored at $(dir)!"

    return plots
end
