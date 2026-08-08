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

# Shared random subset of indices, so every panel of a figure shows the SAME points
#   - with replacement: fine for a scatter, and O(max_pts) instead of randperm's O(n)
#     on the ~10^6 points a full stats run produces
thin_idx(n, max_pts) = n <= max_pts ? (1:n) : rand(1:n, max_pts)


# One scatter panel with its fitted line and the fraction of variance the line explains
#   - `line` is a function of the predictor, so each panel supplies its own fit
function regression_panel!(pos, x, y, line; xlabel, ylabel, title)

    ax = CairoMakie.Axis(pos; xlabel, ylabel, title, titlesize = 12)
    CairoMakie.scatter!(ax, x, y; markersize = 2, alpha = 0.2)

    # Fitted line of the column-mean coefficients
    xr = range(extrema(x)...; length = 2)
    CairoMakie.lines!(ax, xr, line.(xr); color = :red)

    # R² of the column-MEAN line against the pooled scatter — not expected to be near 1,
    # it is meant for comparing output forms (linear vs planck) on the same field
    r2 = 1 - sum(abs2, y .- line.(x)) / sum(abs2, y .- mean(y))
    CairoMakie.text!(ax, 0.03, 0.97; text = "R² = $(round(r2, digits = 3))",
                     space = :relative, align = (:left, :top), fontsize = 10)

    return ax
end


# dT = a + b*P(T), fitted per column and layer — one panel per layer
function plot_regression_dT(samples, lin, form = LinearOutput(); ncols = 4, max_pts = 3000)

    nlayers = Base.size(samples.T, 2)
    fig = CairoMakie.Figure(size = (350*ncols, 300*cld(nlayers, ncols)))

    # Plot against the predictor the coefficients were fitted on
    P = predictor(form, samples.T)

    # One shared subset for every layer, so the panels are comparable
    idx = thin_idx(Base.size(P, 1) * Base.size(P, 3), max_pts)

    for k in 1:nlayers
        r, c = fldmod1(k, ncols)

        # Line uses the column-MEAN coefficients, the scatter pools all columns,
        # so a spread around the line is expected — it is not a bad fit
        a, b = lin.a.mean[k], lin.b.mean[k]

        regression_panel!(fig[r, c], vec(P[:,k,:])[idx], vec(samples.dT[:,k,:])[idx],
            x -> a + b*(x - lin.center.T[k]);
            xlabel = "P(T)", ylabel = "dT (K/s)", title = "Layer $k")
    end

    CairoMakie.Label(fig[0, :],
        "$(nameof(typeof(form))): dT = a + b*(P(T) - P̄)    (red: column-mean a, b)"; fontsize = 17, font = :bold)
    return fig
end


# Both flux regressions in one figure, as PARTIAL residuals:
#   olw = c + d*(P(T[mid]) - P̄mid) + e*(P(Ts) - P̄s)
#     -> panel 1 plots olw minus the e-term, panel 2 olw minus the d-term, so the red
#        line is the only thing left to explain the scatter.
#        Plotting raw olw makes the line look far too shallow: the cloud shows the TOTAL
#        dependence (T[mid] and Ts co-vary strongly), the line only the PARTIAL one.
#   slwd = f + g*(P(T[nlayers]) - P̄[nlayers])    single predictor, nothing to subtract
function plot_regression_flux(samples, lin, form = LinearOutput(); max_pts = 5000)

    nlayers = Base.size(samples.T, 2)
    mid     = mid_layer(nlayers)

    # Predictors the coefficients were fitted on
    #   - transform once: predictor() copies the whole profile array
    P  = predictor(form, samples.T)
    T1 = vec(P[:, mid, :])                      # mid-layer temperature
    Tb = vec(P[:, nlayers, :])                  # bottom-layer temperature
    Ts = vec(predictor(form, samples.Ts))       # surface temperature

    olw  = vec(samples.olw)
    slwd = vec(samples.slwd)

    # Column-mean coefficients and predictor centers — what ConstLW starts from
    c, d, e = lin.c.mean[1], lin.d.mean[1], lin.e.mean[1]
    f, g    = lin.f.mean[1], lin.g.mean[1]
    cT, cTs = lin.center.T, lin.center.Ts[1]

    # One shared subset for every panel, so features can be cross-referenced
    idx = thin_idx(length(olw), max_pts)

    fig = CairoMakie.Figure(size = (1200, 380))

    # (1) olw against mid-layer T, with the Ts contribution removed
    regression_panel!(fig[1,1], T1[idx], (olw .- e .* (Ts .- cTs))[idx],
        x -> c + d*(x - cT[mid]);
        xlabel = "P(T[$mid])", ylabel = "olw - e*(P(Ts)-P̄s)  [W/m²]",
        title  = "olw vs mid-layer T")

    # (2) olw against surface T, with the T[mid] contribution removed
    regression_panel!(fig[1,2], Ts[idx], (olw .- d .* (T1 .- cT[mid]))[idx],
        x -> c + e*(x - cTs);
        xlabel = "P(Ts)", ylabel = "olw - d*(P(T[mid])-P̄mid)  [W/m²]",
        title  = "olw vs surface T")

    # (3) slwd against bottom-layer T — single predictor, nothing to subtract
    regression_panel!(fig[1,3], Tb[idx], slwd[idx],
        x -> f + g*(x - cT[nlayers]);
        xlabel = "P(T[nlayers])", ylabel = "slwd  [W/m²]",
        title  = "slwd vs bottom-layer T")

    CairoMakie.Label(fig[0, :],
        "$(nameof(typeof(form))) — partial residuals, red: column-mean coefficients";
        fontsize = 13, font = :bold)
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
