import re

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pypsa

def cluster_by_bus_map(n):
    groups = n.buses.operator.apply(lambda x: re.split(" |,|;", x)[0])
    busmap = groups.where(groups != "", n.buses.index)

    n.lines = n.lines.reindex(columns=n.components["Line"]["attrs"].index[1:])
    n.lines["type"] = np.nan
    n.buses = n.buses.reindex(columns=n.components["Bus"]["attrs"].index[1:])
    n.buses["frequency"] = 50

    C = n.cluster.get_clustering_from_busmap(busmap)

    nc = C.n

    fig, (ax, ax1) = plt.subplots(
        1, 2, subplot_kw={"projection": ccrs.EqualEarth()}, figsize=(12, 12)
    )
    plot_kwrgs = dict(bus_sizes=1e-3, line_widths=0.5)
    n.plot(ax=ax, title="original", **plot_kwrgs)
    nc.plot(ax=ax1, title="clustered by operator", **plot_kwrgs)
    fig.tight_layout()

    fig.show()


def cluster_by_bus_map_created_from_k_means(n):
    weighting = pd.Series(1, n.buses.index)
    busmap2 = n.cluster.busmap_by_kmeans(bus_weightings=weighting, n_clusters=50)

    nc2 = n.cluster.cluster_by_busmap(busmap2)

    fig, (ax, ax1) = plt.subplots(
        1, 2, subplot_kw={"projection": ccrs.EqualEarth()}, figsize=(12, 12)
    )
    plot_kwrgs = dict(bus_sizes=1e-3, line_widths=0.5)
    n.plot(ax=ax, title="original", **plot_kwrgs)
    nc2.plot(ax=ax1, title="clustered by kmeans", **plot_kwrgs)
    fig.tight_layout()

    fig.show()


n = pypsa.examples.scigrid_de()
n.calculate_dependent_values()

cluster_by_bus_map(n)
# cluster_by_bus_map_created_from_k_means(n) # not working