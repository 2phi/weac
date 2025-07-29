import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def distribution(
    data: pd.Series,
    kind: str = "pdf",
    bins: int = 75,
    plot_range: tuple[float, float] = (0, 25),
    fit_to_range: bool = False,
    density: bool = True,
    histogram: bool = True,
    function: bool = True,
    zorder: int | None = None,
    log: bool = False,
    dist_type: str = "lognorm",
    save: str = None,
):
    """
    Fit and plot the specified distribution (PDF or CDF) for the given data.

    Parameters
    ----------
    data : pd.Series
        Dataset to be analyzed.
    kind : str, optional
        Type of distribution to plot: 'pdf' or 'cdf'. Default is 'pdf'.
    bins : int, optional
        Number of bins for the histogram. Default is 75.
    plot_range : tuple[float, float], optional
        Range for the histogram and plot. Default is (0, 25).
    fit_to_range : bool, optional
        If True, filters data to be within the specified range. Default is False.
    density : bool, optional
        If True, the histogram is normalized to form a probability density.
        Default is True.
    histogram : bool, optional
        Whether to plot the histogram. Default is True.
    function : bool, optional
        Whether to plot the fitted distribution function (PDF or CDF).
        Default is True.
    zorder : int or None, optional
        The drawing order of plot elements. If None, defaults to 2 for 'pdf' and
        1 for 'cdf'. If provided, uses the given value.
    log : bool, optional
        If True, plots with logarithmically spaced x-axes. Default is False.
    dist_type : str, optional
        Type of distribution to fit and plot: 'lognorm', 'cauchy', 'chi2', or 'expon'.
        Default is 'lognorm'.
    save : str, optional
        If provided, saves the plot to a file. Default is None.
    Raises
    ------
    ValueError
        If the 'kind' parameter is not 'pdf' or 'cdf'.
    ValueError
        If the 'dist_type' parameter is not 'lognorm', 'cauchy', 'chi2', or 'expon'.
    TypeError
        If zorder is not an integer or None.

    Examples
    --------
    >>> data = pd.Series(np.random.lognormal(mean=1, sigma=0.5, size=1000))
    >>> lognorm_distribution(data, kind='pdf', log=True, dist_type='lognorm')
    >>> lognorm_distribution(data, kind='cdf', log=True, dist_type='cauchy')
    """

    plt.figure()

    # Set default zorder based on 'kind' if zorder is None
    if zorder is None:
        if kind == "pdf":
            zorder = 2
        elif kind == "cdf":
            zorder = 1
        else:
            raise ValueError("Invalid 'kind' parameter. Must be 'pdf' or 'cdf'.")
    else:
        # Ensure zorder is an integer
        if not isinstance(zorder, int):
            raise TypeError("zorder must be an integer or None.")

    # Unpack range
    x_min = 1e-3 if log and plot_range[0] <= 0 else plot_range[0]
    x_max = plot_range[1]

    # Filter data if necessary
    if fit_to_range:
        data = data[(data >= x_min) & (data <= x_max)]

    # Fit the specified distribution to the data
    if dist_type == "lognorm":
        dist = stats.lognorm
        params = dist.fit(data)
        shape, loc, scale = params
        args = (shape,)
        kwargs = {"loc": loc, "scale": scale}
    elif dist_type == "cauchy":
        dist = stats.cauchy
        params = dist.fit(data)
        loc, scale = params
        args = ()
        kwargs = {"loc": loc, "scale": scale}
    elif dist_type == "expon":
        dist = stats.expon
        params = dist.fit(data)
        loc, scale = params
        args = ()
        kwargs = {"loc": loc, "scale": scale}
    elif dist_type == "chi2":
        dist = stats.chi2
        params = dist.fit(data)
        df, loc, scale = params
        args = (df,)
        kwargs = {"loc": loc, "scale": scale}
    elif dist_type == "beta":
        dist = stats.beta
        params = dist.fit(data)
        a, b, loc, scale = params
        args = (a, b)
        kwargs = {"loc": loc, "scale": scale}
    else:
        raise ValueError(
            "Invalid 'dist_type' parameter. Must be 'lognorm', 'cauchy', 'chi2', or 'expon'."
        )

    # Generate bin edges
    if log:
        bins_edges = np.logspace(np.log10(x_min), np.log10(x_max), bins + 1)
    else:
        bins_edges = np.linspace(x_min, x_max, bins + 1)

    # Compute the histogram
    hist_data, hist_bins = np.histogram(data, bins=bins_edges, density=density)

    # For CDF, compute the cumulative sum of histogram data
    if kind == "cdf":
        # Multiply by bin widths to get probability masses
        hist_data = np.cumsum(hist_data * np.diff(hist_bins))

    # Calculate bin widths
    bar_widths = 0.7 * np.diff(hist_bins)

    # Plot the histogram
    if histogram:
        plt.bar(
            hist_bins[:-1],
            hist_data,
            width=bar_widths,
            color="w",
            zorder=zorder,
            align="center",
        )
        plt.bar(
            hist_bins[:-1],
            hist_data,
            width=bar_widths,
            alpha=0.5,
            zorder=zorder,
            align="center",
        )

    # Generate x values for plotting the function
    if log:
        x = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
    else:
        x = np.linspace(x_min, x_max, 1000)

    # Calculate the PDF or CDF based on the 'kind' parameter
    if kind == "pdf":
        y_data = dist.pdf(x, *args, **kwargs)
    elif kind == "cdf":
        y_data = dist.cdf(x, *args, **kwargs)
    else:
        raise ValueError("Invalid 'kind' parameter. Must be 'pdf' or 'cdf'.")

    # Plot the fitted distribution function
    if function and density:
        if not histogram:
            plt.fill_between(x, y_data, zorder=zorder, alpha=0.8, color="w")
            plt.fill_between(x, y_data, zorder=zorder, alpha=0.2)
        plt.plot(x, y_data, color="w", lw=3, zorder=zorder)
        plt.plot(x, y_data, zorder=zorder, label=dist_type)

    # Set the x-axis to logarithmic scale if log=True
    if log:
        plt.xscale("log")

    plt.xlim(x_min, x_max)
    plt.legend()

    if save:
        plt.savefig(save)
    else:
        plt.show()
