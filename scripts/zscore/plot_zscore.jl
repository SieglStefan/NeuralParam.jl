### Functions for plotting zscore statistics
###
### - Histograms of the z-score normalized inputs and outputs (should look ~N(0,1))
### - Regression diagnostics for the linear output form:
###       dT   = a*T + b              one panel per layer
###       olw  = c + d*T[1] + e*Ts    one panel per predictor
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
    data   = [vec(NeuralParam.zscore(selectdim(x, 2, k), st.mean[k], st.std[k])) for k in 1:nlayers]
    titles = ["Layer $k" for k in 1:nlayers]

    return plot_histograms(data, titles; ncols,
        suptitle = "Normalized $name — per layer",
        size     = (350*ncols, 300*cld(nlayers, ncols)))
end


# Histograms of several normalized SCALAR variables, all in one figure
function plot_scalar_histograms(s, st, names; suptitle = "Normalized scalars", ncols = 3)

    # Look both the samples and their stats up by name
    data   = [vec(NeuralParam.zscore(getproperty(s, n),
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


# dT = a*T + b, fitted per column and layer — one panel per layer
function plot_regression_dT(sub, lin; ncols = 4, max_pts = 3000)

    nlayers = Base.size(sub.T, 2)
    fig = CairoMakie.Figure(size = (350*ncols, 300*cld(nlayers, ncols)))

    for k in 1:nlayers
        r, c = fldmod1(k, ncols)
        ax = CairoMakie.Axis(fig[r, c]; xlabel = "T (K)", ylabel = "dT (K/s)", title = "Layer $k")

        x, y = thin(max_pts, vec(sub.T[:,k,:]), vec(sub.dT[:,k,:]))
        CairoMakie.scatter!(ax, x, y; markersize = 2, alpha = 0.2)

        # Line uses the column-MEAN coefficients, the scatter pools all columns,
        # so a spread around the line is expected — it is not a bad fit
        xr = range(extrema(x)...; length = 2)
        CairoMakie.lines!(ax, xr, lin.a.mean[k] .* xr .+ lin.b.mean[k]; color = :red)
    end

    CairoMakie.Label(fig[0, :], "dT = a*T + b   (red: column-mean a, b)"; fontsize = 17, font = :bold)
    return fig
end


# Both flux regressions in one figure:
#   olw  = c + d*T[1] + e*Ts    two predictors, one panel each with the other held at its mean
#   slwd = f + g*T[nlayers]     one predictor, plain scatter with its line
function plot_regression_flux(sub, lin; max_pts = 5000)

    nlayers = Base.size(sub.T, 2)
    T1 = sub.T[:, 4, :]                     # top-layer temperature
    Tb = sub.T[:, nlayers, :]               # bottom-layer temperature

    # Column-mean coefficients — what ConstLW starts from
    c, d, e = lin.c.mean[1], lin.d.mean[1], lin.e.mean[1]
    f, g    = lin.f.mean[1], lin.g.mean[1]

    fig = CairoMakie.Figure(size = (1050, 340))

    # (1) olw vs T[1], with Ts held at its mean so a line can be drawn
    ax1 = CairoMakie.Axis(fig[1,1]; xlabel = "T[4] (K)", ylabel = "olw (W/m²)",
                          title = "olw vs top-layer T")
    x, y = thin(max_pts, vec(T1), vec(sub.olw))
    CairoMakie.scatter!(ax1, x, y; markersize = 2, alpha = 0.2)
    xr = range(extrema(x)...; length = 2)
    CairoMakie.lines!(ax1, xr, c .+ d .* xr .+ e * mean(sub.Ts); color = :red)

    # (2) olw vs Ts, with T[1] held at its mean
    ax2 = CairoMakie.Axis(fig[1,2]; xlabel = "Ts (K)", ylabel = "olw (W/m²)",
                          title = "olw vs surface T")
    x, y = thin(max_pts, vec(sub.Ts), vec(sub.olw))
    CairoMakie.scatter!(ax2, x, y; markersize = 2, alpha = 0.2)
    xr = range(extrema(x)...; length = 2)
    CairoMakie.lines!(ax2, xr, c .+ d * mean(T1) .+ e .* xr; color = :red)

    # (3) slwd vs T[nlayers], only one predictor
    ax3 = CairoMakie.Axis(fig[1,3]; xlabel = "T[nlayers] (K)", ylabel = "slwd (W/m²)",
                          title = "slwd vs bottom-layer T")
    x, y = thin(max_pts, vec(Tb), vec(sub.slwd))
    CairoMakie.scatter!(ax3, x, y; markersize = 2, alpha = 0.2)
    xr = range(extrema(x)...; length = 2)
    CairoMakie.lines!(ax3, xr, f .+ g .* xr; color = :red)

    CairoMakie.Label(fig[0, :],
        "olw = c + d*T[4] + e*Ts        slwd = f + g*T[nlayers]        (red: column-mean coefficients)";
        fontsize = 15, font = :bold)
    return fig
end
