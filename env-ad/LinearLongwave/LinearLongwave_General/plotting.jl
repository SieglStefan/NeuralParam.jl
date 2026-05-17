
function plot_calibration(L, a, b)

    p1 = plot(
        0:length(L)-1,
        L,
        yscale=:log10,
        xlabel="Training step",
        ylabel="Loss",
        title="Loss evolution",
        label="Loss"
    )

    p2 = plot(
        0:length(a)-1,
        a,
        xlabel="Training step",
        ylabel="a",
        title="Parameter a",
        label="a"
    )

    p3 = plot(
        0:length(b)-1,
        b,
        xlabel="Training step",
        ylabel="b",
        title="Parameter b",
        label="b"
    )

    plot(p1, p2, p3, layout=(3,1))
end