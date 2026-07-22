import argparse
import time
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


LATTICE_BASIS = {
    'scc': np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    'bcc': np.array([[0.0, 0.0, 0.0],
                     [0.5, 0.5, 0.5]], dtype=np.float64),
    'fcc': np.array([[0.0, 0.0, 0.0],
                     [0.0, 0.5, 0.5],
                     [0.5, 0.0, 0.5],
                     [0.5, 0.5, 0.0]], dtype=np.float64),
}

NEAREST_NEIGHBOR_DISTANCE = {
    'scc': 1.0,
    'bcc': np.sqrt(3.0) / 2.0,
    'fcc': 1.0 / np.sqrt(2.0),
}

FIRST_SHELL_COORDINATION = {
    'scc': 6.0,
    'bcc': 8.0,
    'fcc': 12.0,
}

# Cut at the midpoint between the first and second ideal neighbor shells.
FIRST_SHELL_CUTOFF_OVER_NN = {
    'scc': 0.5 * (1.0 + np.sqrt(2.0)),
    'bcc': 0.5 * (1.0 + 2.0 / np.sqrt(3.0)),
    'fcc': 0.5 * (1.0 + np.sqrt(2.0)),
}

DISPLAY_LABELS = {'bcc': 'BCC', 'fcc': 'FCC', 'scc': 'SC'}


def default_factors():
    return np.array([
        0.00, 0.02, 0.04, 0.06, 0.08, 0.10,
        0.12, 0.14, 0.16, 0.18, 0.20, 0.22,
        0.24, 0.26, 0.28, 0.30, 0.35, 0.40,
        0.45, 0.50,
    ], dtype=np.float64)


def make_periodic_lattice(lattice, cells):
    basis = LATTICE_BASIS[lattice]
    positions = []
    for i in range(cells):
        for j in range(cells):
            for k in range(cells):
                cell_origin = np.array([i, j, k], dtype=np.float64)
                positions.append(cell_origin + basis)
    return np.concatenate(positions, axis=0)


def apply_vibration(positions, box_length, nearest_neighbor_distance, factor, rng):
    sigma = factor * nearest_neighbor_distance / np.sqrt(3.0)
    displacement = rng.normal(0.0, sigma, size=positions.shape)
    displaced = np.mod(positions + displacement, box_length)
    measured_l = np.sqrt(np.mean(np.sum(displacement * displacement, axis=1))) / nearest_neighbor_distance
    return displaced, measured_l


def pair_distance_histogram_pbc(positions, box_length, edges):
    hist = np.zeros(len(edges) - 1, dtype=np.float64)
    n_atom = positions.shape[0]
    for i in range(n_atom - 1):
        delta = positions[i + 1:] - positions[i]
        delta -= box_length * np.rint(delta / box_length)
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        hist += np.histogram(distances, bins=edges)[0]
    return hist


def normalize_rdf(pair_hist, n_atom, box_length, shell_edges, n_samples):
    volume = box_length ** 3
    rho = n_atom / volume
    shell_volume = (4.0 * np.pi / 3.0) * (shell_edges[1:] ** 3 - shell_edges[:-1] ** 3)
    # pair_hist counts each pair once; RDF normalization is for ordered central-neighbor pairs.
    return 2.0 * pair_hist / (n_samples * n_atom * rho * shell_volume)


def estimate_fwhm(r_over_nn, gr):
    peak_window = (r_over_nn >= 0.65) & (r_over_nn <= 1.35)
    if not np.any(peak_window):
        return np.nan, np.nan, np.nan

    indices = np.where(peak_window)[0]
    peak_idx = indices[np.argmax(gr[indices])]
    peak_height = gr[peak_idx]
    peak_position = r_over_nn[peak_idx]
    if not np.isfinite(peak_height) or peak_height <= 0.0:
        return peak_position, peak_height, np.nan

    half_height = 0.5 * peak_height
    left_idx = peak_idx
    while left_idx > 0 and gr[left_idx] >= half_height:
        left_idx -= 1
    right_idx = peak_idx
    while right_idx < len(gr) - 1 and gr[right_idx] >= half_height:
        right_idx += 1

    if left_idx == peak_idx or right_idx == peak_idx:
        return peak_position, peak_height, np.nan
    return peak_position, peak_height, r_over_nn[right_idx] - r_over_nn[left_idx]


def coordination_to_cutoff(r_over_nn, gr, rho_scaled, cutoff_over_nn):
    mask = r_over_nn <= cutoff_over_nn
    if not np.any(mask):
        return np.nan
    integrand = 4.0 * np.pi * rho_scaled * r_over_nn[mask] ** 2 * gr[mask]
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(integrand, r_over_nn[mask])
    return np.trapz(integrand, r_over_nn[mask])


def evaluate_lattice(lattice, factors, cells, n_samples, r_max_over_nn, n_bins, seed, output_dir):
    rng = np.random.default_rng(seed)
    positions0 = make_periodic_lattice(lattice, cells)
    n_atom = positions0.shape[0]
    box_length = float(cells)
    d0 = float(NEAREST_NEIGHBOR_DISTANCE[lattice])
    rho_scaled = (n_atom / box_length ** 3) * d0 ** 3

    r_edges_over_nn = np.linspace(0.0, r_max_over_nn, n_bins + 1, dtype=np.float64)
    r_centers_over_nn = 0.5 * (r_edges_over_nn[1:] + r_edges_over_nn[:-1])
    r_edges = r_edges_over_nn * d0

    gr = np.zeros((len(factors), n_bins), dtype=np.float64)
    measured_lindemann_mean = np.zeros(len(factors), dtype=np.float64)
    measured_lindemann_std = np.zeros(len(factors), dtype=np.float64)
    first_peak_position = np.zeros(len(factors), dtype=np.float64)
    first_peak_height = np.zeros(len(factors), dtype=np.float64)
    first_peak_fwhm = np.zeros(len(factors), dtype=np.float64)
    first_shell_coordination = np.zeros(len(factors), dtype=np.float64)

    for factor_index, factor in enumerate(factors):
        t0 = time.time()
        pair_hist = np.zeros(n_bins, dtype=np.float64)
        measured = np.zeros(n_samples, dtype=np.float64)

        for sample_index in range(n_samples):
            displaced, measured_l = apply_vibration(positions0, box_length, d0, factor, rng)
            pair_hist += pair_distance_histogram_pbc(displaced, box_length, r_edges)
            measured[sample_index] = measured_l

        gr_factor = normalize_rdf(pair_hist, n_atom, box_length, r_edges, n_samples)
        gr[factor_index] = gr_factor
        measured_lindemann_mean[factor_index] = measured.mean()
        measured_lindemann_std[factor_index] = measured.std()

        peak_pos, peak_height, peak_fwhm = estimate_fwhm(r_centers_over_nn, gr_factor)
        first_peak_position[factor_index] = peak_pos
        first_peak_height[factor_index] = peak_height
        first_peak_fwhm[factor_index] = peak_fwhm
        first_shell_coordination[factor_index] = coordination_to_cutoff(
            r_centers_over_nn,
            gr_factor,
            rho_scaled,
            FIRST_SHELL_CUTOFF_OVER_NN[lattice],
        )

        print(
            lattice,
            'factor', float(factor),
            'measured_L', float(measured_lindemann_mean[factor_index]),
            'first_peak_height', float(first_peak_height[factor_index]),
            'first_peak_fwhm', float(first_peak_fwhm[factor_index]),
            'first_shell_coordination', float(first_shell_coordination[factor_index]),
            'time', time.time() - t0,
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'rdf-{lattice}-vibration-sweep.npz'
    np.savez(
        out_path,
        lattice=np.array(lattice),
        factors=factors,
        measured_lindemann_mean=measured_lindemann_mean,
        measured_lindemann_std=measured_lindemann_std,
        r_over_nn=r_centers_over_nn,
        gr=gr,
        first_peak_position=first_peak_position,
        first_peak_height=first_peak_height,
        first_peak_fwhm=first_peak_fwhm,
        first_shell_coordination=first_shell_coordination,
        first_shell_coordination_ideal=np.array(FIRST_SHELL_COORDINATION[lattice], dtype=np.float64),
        first_shell_cutoff_over_nn=np.array(FIRST_SHELL_CUTOFF_OVER_NN[lattice], dtype=np.float64),
        nearest_neighbor_distance=np.array(d0, dtype=np.float64),
        cells=np.array(cells, dtype=np.int64),
        n_atom=np.array(n_atom, dtype=np.int64),
        n_samples=np.array(n_samples, dtype=np.int64),
        r_max_over_nn=np.array(r_max_over_nn, dtype=np.float64),
        n_bins=np.array(n_bins, dtype=np.int64),
        seed=np.array(seed, dtype=np.int64),
        rdf_geometry=np.array('periodic_cubic_supercell'),
        distance_axis=np.array('r_over_clean_nearest_neighbor_distance'),
    )
    print('saved', out_path)
    return out_path


def load_results(paths):
    results = {}
    for path in paths:
        data = np.load(path, allow_pickle=True)
        lattice = str(data['lattice'])
        results[lattice] = {key: data[key] for key in data.files}
        data.close()
    return results


def select_plot_indices(factors, requested):
    indices = []
    for value in requested:
        indices.append(int(np.argmin(np.abs(factors - value))))
    return sorted(set(indices))


def plot_gr_curves(results, output_dir, plot_factors):
    if plt is None:
        print('matplotlib is not installed; skipping RDF plots')
        return None

    lattices = [lattice for lattice in ('bcc', 'fcc', 'scc') if lattice in results]
    fig, axes = plt.subplots(1, len(lattices), figsize=(5.0 * len(lattices), 4.2), sharey=True)
    if len(lattices) == 1:
        axes = [axes]

    cmap = plt.get_cmap('viridis')
    for ax, lattice in zip(axes, lattices):
        result = results[lattice]
        factors = result['factors']
        measured = result['measured_lindemann_mean']
        r = result['r_over_nn']
        indices = select_plot_indices(factors, plot_factors)
        for color_index, factor_index in enumerate(indices):
            color = cmap(color_index / max(1, len(indices) - 1))
            ax.plot(
                r,
                result['gr'][factor_index],
                color=color,
                linewidth=1.8,
                label=f"L={measured[factor_index]:.2f}",
            )
        ax.axvline(1.0, color='black', linestyle=':', linewidth=1.0)
        ax.set_title(DISPLAY_LABELS.get(lattice, lattice.upper()))
        ax.set_xlabel(r'$r / r_{\rm nn}^{0}$')
        ax.set_xlim(0.0, result['r_max_over_nn'])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r'$g(r)$')
    axes[-1].legend(title='measured', fontsize=8)
    fig.tight_layout()

    path = output_dir / 'rdf-gr-curves.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print('saved', path)
    return path


def plot_first_peak_summary(results, output_dir):
    if plt is None:
        print('matplotlib is not installed; skipping first-peak summary plot')
        return None

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.5), sharex=True)
    colors = {'bcc': 'tab:blue', 'fcc': 'tab:green', 'scc': 'tab:red'}
    labels = DISPLAY_LABELS

    for lattice in ('bcc', 'fcc', 'scc'):
        if lattice not in results:
            continue
        result = results[lattice]
        x = result['measured_lindemann_mean']
        axes[0].plot(x, result['first_peak_height'], 'o-', color=colors[lattice], label=labels[lattice])
        axes[1].plot(x, result['first_peak_fwhm'], 'o-', color=colors[lattice], label=labels[lattice])
        axes[2].plot(x, result['first_shell_coordination'], 'o-', color=colors[lattice], label=labels[lattice])
        axes[2].axhline(
            float(result['first_shell_coordination_ideal']),
            color=colors[lattice],
            linestyle=':',
            linewidth=1.0,
            alpha=0.7,
        )

    axes[0].set_ylabel('first peak height')
    axes[1].set_ylabel(r'first peak FWHM in $r/r_{\rm nn}^{0}$')
    axes[2].set_ylabel('coordination to first cutoff')
    axes[2].set_xlabel('measured Lindemann ratio')
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()

    path = output_dir / 'rdf-first-peak-summary.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print('saved', path)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Radial distribution function g(r) sweep for vibrated BCC/FCC/SC periodic cubic lattices.'
    )
    parser.add_argument('--lattices', '--lattice', nargs='+', default=['bcc', 'fcc', 'scc'], choices=['bcc', 'fcc', 'scc'])
    parser.add_argument('--factors', nargs='*', type=float, default=None)
    parser.add_argument('--cells', type=int, default=6, help='Number of conventional cubic cells per side.')
    parser.add_argument('--n-samples', type=int, default=8, help='Independent vibration realizations per factor.')
    parser.add_argument('--r-max-over-nn', type=float, default=3.0)
    parser.add_argument('--n-bins', type=int, default=240)
    parser.add_argument('--seed', type=int, default=20260529)
    parser.add_argument('--output-dir', default='rdf-vibration-results')
    parser.add_argument('--plot-factors', nargs='*', type=float, default=[0.0, 0.10, 0.14, 0.20, 0.30, 0.50])
    parser.add_argument('--no-plots', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    factors = default_factors() if args.factors is None else np.array(args.factors, dtype=np.float64)
    output_dir = Path(args.output_dir)

    result_paths = []
    for offset, lattice in enumerate(args.lattices):
        result_paths.append(
            evaluate_lattice(
                lattice=lattice,
                factors=factors,
                cells=args.cells,
                n_samples=args.n_samples,
                r_max_over_nn=args.r_max_over_nn,
                n_bins=args.n_bins,
                seed=args.seed + 10000 * offset,
                output_dir=output_dir,
            )
        )

    if not args.no_plots:
        results = load_results(result_paths)
        plot_gr_curves(results, output_dir, args.plot_factors)
        plot_first_peak_summary(results, output_dir)


if __name__ == '__main__':
    main()
