import argparse
import importlib.util
import time
from pathlib import Path

try:
    import keras
except ModuleNotFoundError:
    from tensorflow import keras
import numpy as np
from numba import njit

NDIM = 32
NCHANNEL = 1
NMAX = 1000
CLASS_NAMES = ['bcc', 'fcc', 'scc', 'random']
LATTICE_CLASS = {'bcc': 0, 'fcc': 1, 'scc': 2}
RANDOM_CLASS = 3
GENERATOR_FILES = {
    'bcc': 'bcc-traindata.py',
    'fcc': 'fcc-traindata.py',
    'scc': 'scc-traindata.py',
}


def default_factors():
    factors = []
    factor = -0.02
    for k in range(18):
        stepsize = 0.02
        if k > 6:
            stepsize = 0.04
        if k > 12:
            stepsize = 0.08
        factor += stepsize
        factors.append(factor)
    return np.array(factors, dtype=np.float32)


def import_generator(lattice):
    path = Path(GENERATOR_FILES[lattice])
    spec = importlib.util.spec_from_file_location(f'{lattice}_generator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_inner_buffer_indices(x_inner, y_inner, z_inner, n_inner, x_buffer, y_buffer, z_buffer, n_buffer, tol=1.0e-7):
    indices = np.full(n_inner, -1, dtype=np.int64)
    tol2 = tol * tol
    for i in range(n_inner):
        best = -1
        best_d2 = 1.0e30
        for j in range(n_buffer):
            dx = float(x_inner[i] - x_buffer[j])
            dy = float(y_inner[i] - y_buffer[j])
            dz = float(z_inner[i] - z_buffer[j])
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best_d2:
                best_d2 = d2
                best = j
        if best_d2 <= tol2:
            indices[i] = best
    return indices


def apply_buffer_displacement(x, y, z, n, sigma):
    xd = np.array(x, copy=True)
    yd = np.array(y, copy=True)
    zd = np.array(z, copy=True)
    dx = np.zeros(n, dtype=np.float32)
    dy = np.zeros(n, dtype=np.float32)
    dz = np.zeros(n, dtype=np.float32)
    for i in range(n):
        dx[i] = np.random.normal(0.0, sigma)
        dy[i] = np.random.normal(0.0, sigma)
        dz[i] = np.random.normal(0.0, sigma)
        xd[i] = x[i] + dx[i]
        yd[i] = y[i] + dy[i]
        zd[i] = z[i] + dz[i]
    return xd, yd, zd, dx, dy, dz


def inner_buffer_nearest_neighbor_cv(x_inner, y_inner, z_inner, n_inner, x_buffer, y_buffer, z_buffer, n_buffer, inner_buffer_indices=None):
    if n_inner < 1 or n_buffer < 2:
        return 0.0, 0.0, 0.0
    distances = np.zeros(n_inner, dtype=np.float32)
    for i in range(n_inner):
        skip_idx = -1
        if inner_buffer_indices is not None:
            skip_idx = int(inner_buffer_indices[i])
        best = 1.0e30
        for j in range(n_buffer):
            if j == skip_idx:
                continue
            dx = float(x_inner[i] - x_buffer[j])
            dy = float(y_inner[i] - y_buffer[j])
            dz = float(z_inner[i] - z_buffer[j])
            d2 = dx * dx + dy * dy + dz * dz
            if d2 > 1.0e-14 and d2 < best:
                best = d2
        distances[i] = np.sqrt(best)
    mean = float(distances.mean())
    std = float(distances.std())
    cv = 0.0
    if mean > 0.0:
        cv = std / mean
    return mean, std, cv


def rms_displacement_for_indices(dx, dy, dz, indices):
    valid = indices[indices >= 0]
    if valid.size == 0:
        return 0.0
    sq = dx[valid] * dx[valid] + dy[valid] * dy[valid] + dz[valid] * dz[valid]
    return float(np.sqrt(sq.mean()))


def entropy(probabilities, eps=1.0e-8):
    p = np.clip(probabilities, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def sample_clean_lattice(generator, rng_seed=None):
    # Periodic-supercell generators return a complete periodic atom set directly.
    return generator.true_lattice_with_metadata(generator.NMAX), 1


def nearest_neighbor_reference_from_positions(x_inner, y_inner, z_inner, n_inner, x_buffer, y_buffer, z_buffer, n_buffer, inner_buffer_indices):
    mean, _, _ = inner_buffer_nearest_neighbor_cv(
        x_inner, y_inner, z_inner, n_inner,
        x_buffer, y_buffer, z_buffer, n_buffer,
        inner_buffer_indices,
    )
    return mean


def unpack_lattice_output(lattice, out):
    # Current cubic generators return a uniform quaternion after lattice spacing.
    # Older generators returned three Euler angles there. The sweep does not use
    # rotation parameters, so accept both tuple shapes for compatibility.
    if len(out) == 14:
        x, y, z, n, x_inner, y_inner, z_inner, n_inner, a, qw, qx, qy, qz, scaling = out
    else:
        x, y, z, n, x_inner, y_inner, z_inner, n_inner, a, th1, th2, th3, scaling = out
    c_over_a = np.nan
    return x, y, z, n, x_inner, y_inner, z_inner, n_inner, float(a), float(c_over_a), int(scaling)


def evaluate_lattice(model, lattice, factors, nval, batch_size, seed, output_dir):
    generator = import_generator(lattice)
    np.random.seed(seed)
    generator.seed_numba_rng(seed)

    lattice_idx = LATTICE_CLASS[lattice]
    n_factor = len(factors)

    same_lattice_acc = np.zeros(n_factor, dtype=np.float32)
    solid_acc = np.zeros(n_factor, dtype=np.float32)
    lattice_prob_mean = np.zeros(n_factor, dtype=np.float32)
    lattice_prob_std = np.zeros(n_factor, dtype=np.float32)
    solid_prob_mean = np.zeros(n_factor, dtype=np.float32)
    solid_prob_std = np.zeros(n_factor, dtype=np.float32)
    entropy_mean = np.zeros(n_factor, dtype=np.float32)
    atom_count_mean = np.zeros(n_factor, dtype=np.float32)
    atom_count_std = np.zeros(n_factor, dtype=np.float32)
    measured_lindemann_mean = np.zeros(n_factor, dtype=np.float32)
    measured_lindemann_std = np.zeros(n_factor, dtype=np.float32)
    nn_cv_mean = np.zeros(n_factor, dtype=np.float32)
    nn_cv_std = np.zeros(n_factor, dtype=np.float32)
    pred_class_counts = np.zeros((n_factor, len(CLASS_NAMES)), dtype=np.int64)

    for k, factor in enumerate(factors):
        t0 = time.time()
        same_count = 0
        solid_count = 0
        all_lattice_prob = []
        all_solid_prob = []
        all_entropy = []
        all_atom_count = []
        all_measured_lindemann = []
        all_nn_cv = []

        done = 0
        while done < nval:
            current = min(batch_size, nval - done)
            xbatch = np.zeros((current, NDIM, NDIM, NDIM, NCHANNEL), dtype=np.float32)
            batch_atom_count = np.zeros(current, dtype=np.float32)
            batch_measured = np.zeros(current, dtype=np.float32)
            batch_nn_cv = np.zeros(current, dtype=np.float32)

            for i in range(current):
                out, attempts = sample_clean_lattice(generator)
                x, y, z, n, x_inner, y_inner, z_inner, n_inner, a, c_over_a, scaling = unpack_lattice_output(lattice, out)
                inner_indices = find_inner_buffer_indices(x_inner, y_inner, z_inner, n_inner, x, y, z, n)
                d0 = nearest_neighbor_reference_from_positions(
                    x_inner, y_inner, z_inner, n_inner,
                    x, y, z, n,
                    inner_indices,
                )
                # Treat factor as target Lindemann ratio: RMS displacement / nearest-neighbor distance.
                # Component sigma is divided by sqrt(3) so the 3D RMS displacement is factor*d0.
                sigma = factor * d0 / np.sqrt(3.0)
                xd, yd, zd, dx, dy, dz = apply_buffer_displacement(x, y, z, n, sigma)
                xdi = xd[inner_indices]
                ydi = yd[inner_indices]
                zdi = zd[inner_indices]
                image = generator.image_create(NDIM, n, xd, yd, zd)
                xbatch[i] = image.reshape(NDIM, NDIM, NDIM, NCHANNEL)
                batch_atom_count[i] = n_inner
                rms = rms_displacement_for_indices(dx, dy, dz, inner_indices)
                if d0 > 0:
                    batch_measured[i] = rms / d0
                _, _, cv = inner_buffer_nearest_neighbor_cv(xdi, ydi, zdi, n_inner, xd, yd, zd, n, inner_indices)
                batch_nn_cv[i] = cv

            probs = model.predict(xbatch, batch_size=batch_size, verbose=0)
            pred = probs.argmax(axis=1)
            same_count += int((pred == lattice_idx).sum())
            solid_count += int((pred != RANDOM_CLASS).sum())
            pred_class_counts[k] += np.bincount(pred, minlength=len(CLASS_NAMES))

            lattice_prob = probs[:, lattice_idx]
            solid_prob = probs[:, :RANDOM_CLASS].sum(axis=1)
            all_lattice_prob.append(lattice_prob)
            all_solid_prob.append(solid_prob)
            all_entropy.append(entropy(probs))
            all_atom_count.append(batch_atom_count)
            all_measured_lindemann.append(batch_measured)
            all_nn_cv.append(batch_nn_cv)
            done += current

        lp = np.concatenate(all_lattice_prob)
        sp = np.concatenate(all_solid_prob)
        ent = np.concatenate(all_entropy)
        ac = np.concatenate(all_atom_count)
        ml = np.concatenate(all_measured_lindemann)
        cv = np.concatenate(all_nn_cv)

        same_lattice_acc[k] = same_count / nval
        solid_acc[k] = solid_count / nval
        lattice_prob_mean[k] = lp.mean()
        lattice_prob_std[k] = lp.std()
        solid_prob_mean[k] = sp.mean()
        solid_prob_std[k] = sp.std()
        entropy_mean[k] = ent.mean()
        atom_count_mean[k] = ac.mean()
        atom_count_std[k] = ac.std()
        measured_lindemann_mean[k] = ml.mean()
        measured_lindemann_std[k] = ml.std()
        nn_cv_mean[k] = cv.mean()
        nn_cv_std[k] = cv.std()

        print(
            lattice,
            'factor', float(factor),
            'same_lattice_acc', float(same_lattice_acc[k]),
            'solid_acc', float(solid_acc[k]),
            'P_lattice', float(lattice_prob_mean[k]),
            'P_solid', float(solid_prob_mean[k]),
            'measured_L', float(measured_lindemann_mean[k]),
            'nn_cv', float(nn_cv_mean[k]),
            'time', time.time() - t0,
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'combined-{lattice}-vibration-sweep.npz'
    np.savez(
        out_path,
        lattice=np.array(lattice),
        class_names=np.array(CLASS_NAMES),
        factors=factors,
        same_lattice_acc=same_lattice_acc,
        solid_acc=solid_acc,
        lattice_prob_mean=lattice_prob_mean,
        lattice_prob_std=lattice_prob_std,
        solid_prob_mean=solid_prob_mean,
        solid_prob_std=solid_prob_std,
        entropy_mean=entropy_mean,
        atom_count_mean=atom_count_mean,
        atom_count_std=atom_count_std,
        measured_lindemann_mean=measured_lindemann_mean,
        measured_lindemann_std=measured_lindemann_std,
        nn_cv_mean=nn_cv_mean,
        nn_cv_std=nn_cv_std,
        pred_class_counts=pred_class_counts,
        nval=np.array(nval, dtype=np.int64),
        batch_size=np.array(batch_size, dtype=np.int64),
        seed=np.array(seed, dtype=np.int64),
    )
    print('saved', out_path)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description='Combined-model vibration/Lindemann sweep for BCC/FCC/SCC.')
    parser.add_argument('--model', default='saved-model-combined-bcc-fcc-scc-random-best.keras')
    parser.add_argument('--lattices', '--lattice', nargs='+', default=['bcc', 'fcc', 'scc'], choices=['bcc', 'fcc', 'scc'])
    parser.add_argument('--nval', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20260521)
    parser.add_argument('--output-dir', default='combined-vibration-results')
    parser.add_argument('--factors', nargs='*', type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    factors = default_factors() if args.factors is None else np.array(args.factors, dtype=np.float32)
    model = keras.models.load_model(args.model)
    output_dir = Path(args.output_dir)
    for offset, lattice in enumerate(args.lattices):
        evaluate_lattice(
            model=model,
            lattice=lattice,
            factors=factors,
            nval=args.nval,
            batch_size=args.batch_size,
            seed=args.seed + 10000 * offset,
            output_dir=output_dir,
        )


if __name__ == '__main__':
    main()
