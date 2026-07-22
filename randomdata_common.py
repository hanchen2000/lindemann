import os
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

NDIM = 32
NCHANNEL = 1
NMAX = 5000
NPRINT = 1000
LATTICE_NAME = 'Random'
RANDOM_SOURCE_LABEL = np.int8(1)
TARGET_ATOM_COUNT_MIN = 25
TARGET_ATOM_COUNT_MAX = 60
MAX_RANDOM_ATTEMPTS = 200
BUFFER_MARGIN = np.float32(0.35)
BUFFER_VOLUME = float((1.0 + 2.0 * BUFFER_MARGIN) ** 3)
POISSON_LAMBDA_INNER = 42.5
CLUSTER_COUNT_MIN = 4
CLUSTER_COUNT_MAX = 24
CLUSTER_SIGMA_MIN = 0.035
CLUSTER_SIGMA_MAX = 0.095
POINT_REJECTION_ATTEMPTS = 200

RANDOM_GAUSSIAN_CLUSTERED = 0
RANDOM_POISSON_UNIFORM = 1
RANDOM_POISSON_GAUSSIAN = 2
RANDOM_TYPE_ALPHA = np.array([4.0, 4.0, 4.0], dtype=np.float64)


@njit
def seed_numba_rng(seed_value):
    np.random.seed(seed_value)


@njit
def count_inner_atoms(x, y, z, npt):
    count = 0
    for i in range(npt):
        if 0.0 <= x[i] <= 1.0 and 0.0 <= y[i] <= 1.0 and 0.0 <= z[i] <= 1.0:
            count += 1
    return count


@njit
def uniform_random_exact_inner_count(n_inner, n_buffer, nmax, margin):
    x_pos = np.zeros(nmax, dtype=np.float32)
    y_pos = np.zeros(nmax, dtype=np.float32)
    z_pos = np.zeros(nmax, dtype=np.float32)
    width = 1.0 + 2.0 * margin
    n_buffer = min(n_buffer, nmax)
    n_inner = min(n_inner, n_buffer)

    for i in range(n_inner):
        x_pos[i] = np.random.rand()
        y_pos[i] = np.random.rand()
        z_pos[i] = np.random.rand()

    for i in range(n_inner, n_buffer):
        while True:
            x = np.random.rand() * width - margin
            y = np.random.rand() * width - margin
            z = np.random.rand() * width - margin
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= z <= 1.0):
                x_pos[i] = x
                y_pos[i] = y
                z_pos[i] = z
                break

    return x_pos, y_pos, z_pos, n_buffer


@njit
def gaussian_clustered_exact_inner_count(n_inner, n_buffer, nmax, ncluster, sigma, margin):
    x_pos = np.zeros(nmax, dtype=np.float32)
    y_pos = np.zeros(nmax, dtype=np.float32)
    z_pos = np.zeros(nmax, dtype=np.float32)
    width = 1.0 + 2.0 * margin
    n_buffer = min(n_buffer, nmax)
    n_inner = min(n_inner, n_buffer)

    cx_inner = np.zeros(ncluster, dtype=np.float32)
    cy_inner = np.zeros(ncluster, dtype=np.float32)
    cz_inner = np.zeros(ncluster, dtype=np.float32)
    cx_shell = np.zeros(ncluster, dtype=np.float32)
    cy_shell = np.zeros(ncluster, dtype=np.float32)
    cz_shell = np.zeros(ncluster, dtype=np.float32)

    for i in range(ncluster):
        cx_inner[i] = np.random.rand()
        cy_inner[i] = np.random.rand()
        cz_inner[i] = np.random.rand()

        while True:
            x = np.random.rand() * width - margin
            y = np.random.rand() * width - margin
            z = np.random.rand() * width - margin
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= z <= 1.0):
                cx_shell[i] = x
                cy_shell[i] = y
                cz_shell[i] = z
                break

    for i in range(n_inner):
        cidx = int(np.random.rand() * ncluster)
        if cidx >= ncluster:
            cidx = ncluster - 1
        accepted = False
        for _ in range(POINT_REJECTION_ATTEMPTS):
            x = cx_inner[cidx] + np.random.normal(0.0, sigma)
            y = cy_inner[cidx] + np.random.normal(0.0, sigma)
            z = cz_inner[cidx] + np.random.normal(0.0, sigma)
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= z <= 1.0:
                x_pos[i] = x
                y_pos[i] = y
                z_pos[i] = z
                accepted = True
                break
        if not accepted:
            x_pos[i] = np.random.rand()
            y_pos[i] = np.random.rand()
            z_pos[i] = np.random.rand()

    for i in range(n_inner, n_buffer):
        cidx = int(np.random.rand() * ncluster)
        if cidx >= ncluster:
            cidx = ncluster - 1
        accepted = False
        for _ in range(POINT_REJECTION_ATTEMPTS):
            x = cx_shell[cidx] + np.random.normal(0.0, sigma)
            y = cy_shell[cidx] + np.random.normal(0.0, sigma)
            z = cz_shell[cidx] + np.random.normal(0.0, sigma)
            in_buffer = -margin <= x <= 1.0 + margin and -margin <= y <= 1.0 + margin and -margin <= z <= 1.0 + margin
            in_inner = 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= z <= 1.0
            if in_buffer and not in_inner:
                x_pos[i] = x
                y_pos[i] = y
                z_pos[i] = z
                accepted = True
                break
        if not accepted:
            while True:
                x = np.random.rand() * width - margin
                y = np.random.rand() * width - margin
                z = np.random.rand() * width - margin
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= z <= 1.0):
                    x_pos[i] = x
                    y_pos[i] = y
                    z_pos[i] = z
                    break

    return x_pos, y_pos, z_pos, n_buffer


@njit
def light_create(a, b, c, x, y, z, npt):
    light = 0.0
    decay = -200.0
    for i in range(npt):
        dis = (a - x[i]) ** 2 + (b - y[i]) ** 2 + (c - z[i]) ** 2
        light += np.exp(decay * dis)
    return light


@njit(parallel=True)
def image_create(ndim, npt, x, y, z):
    image = np.zeros((ndim, ndim, ndim), dtype=np.float32)
    step = 1.0 / ndim
    for i in prange(ndim):
        for j in range(ndim):
            for k in range(ndim):
                image[i, j, k] = light_create(i * step, j * step, k * step, x, y, z, npt)
    image -= image.min()
    max_value = image.max()
    if max_value > 0.0:
        image /= max_value
    return image


def assert_outputs_writable(paths):
    for path in paths:
        target = Path(path)
        if target.exists():
            if not target.is_file():
                raise RuntimeError(f"Output path exists but is not a regular file: {target}")
            if not os.access(target, os.W_OK):
                raise PermissionError(
                    f"Output file is not writable: {target}. "
                    "Rename/remove it or change ownership before regenerating data."
                )
        else:
            parent = target.parent if str(target.parent) else Path('.')
            if not os.access(parent, os.W_OK):
                raise PermissionError(f"Output directory is not writable: {parent}")


def choose_random_type(random_type_probs):
    r = np.random.rand()
    cumulative = np.cumsum(random_type_probs)
    for idx, cutoff in enumerate(cumulative):
        if r <= cutoff:
            return idx
    return len(random_type_probs) - 1


def bounded_poisson_count():
    for _ in range(100):
        count = int(np.random.poisson(POISSON_LAMBDA_INNER))
        if TARGET_ATOM_COUNT_MIN <= count <= TARGET_ATOM_COUNT_MAX:
            return count
    return int(np.clip(count, TARGET_ATOM_COUNT_MIN, TARGET_ATOM_COUNT_MAX))


def choose_inner_count(random_type):
    if random_type in (RANDOM_POISSON_UNIFORM, RANDOM_POISSON_GAUSSIAN):
        return bounded_poisson_count()
    return int(np.random.randint(TARGET_ATOM_COUNT_MIN, TARGET_ATOM_COUNT_MAX + 1))


def buffer_count_from_inner(inner_count, random_type):
    expected = max(1.0, inner_count * BUFFER_VOLUME)
    return int(max(inner_count, np.random.poisson(expected)))


def choose_ncluster(n_buffer):
    max_cluster = min(CLUSTER_COUNT_MAX, max(CLUSTER_COUNT_MIN, n_buffer))
    return int(np.random.randint(CLUSTER_COUNT_MIN, max_cluster + 1))


def choose_cluster_sigma():
    return float(np.random.uniform(CLUSTER_SIGMA_MIN, CLUSTER_SIGMA_MAX))


def sample_random_positions(random_type):
    for attempt in range(MAX_RANDOM_ATTEMPTS):
        target_inner = choose_inner_count(random_type)
        n_buffer = buffer_count_from_inner(target_inner, random_type)
        ncluster = 0
        sigma = 0.0

        if random_type == RANDOM_POISSON_UNIFORM:
            x, y, z, n = uniform_random_exact_inner_count(target_inner, n_buffer, NMAX, BUFFER_MARGIN)
        else:
            ncluster = choose_ncluster(n_buffer)
            sigma = choose_cluster_sigma()
            x, y, z, n = gaussian_clustered_exact_inner_count(
                target_inner,
                n_buffer,
                NMAX,
                ncluster,
                sigma,
                BUFFER_MARGIN,
            )

        actual_inner = count_inner_atoms(x, y, z, n)
        if TARGET_ATOM_COUNT_MIN <= actual_inner <= TARGET_ATOM_COUNT_MAX:
            return x, y, z, n, target_inner, actual_inner, ncluster, sigma, attempt + 1

    raise RuntimeError('Could not generate random sample in target visible atom-count range')


def make_metadata_arrays(nsample):
    return {
        'target_atom_count': np.zeros(nsample, dtype=np.int32),
        'atom_count': np.zeros(nsample, dtype=np.int32),
        'buffer_atom_count': np.zeros(nsample, dtype=np.int32),
        'random_type': np.zeros(nsample, dtype=np.int8),
        'ncluster': np.zeros(nsample, dtype=np.int32),
        'cluster_sigma': np.zeros(nsample, dtype=np.float32),
        'random_attempts': np.zeros(nsample, dtype=np.int32),
    }


def fill_random_sample(xdata, ydata, index, random_type_probs, metadata_arrays):
    rtype = choose_random_type(random_type_probs)
    x, y, z, n, target_inner, actual_inner, ncluster, sigma, attempts = sample_random_positions(rtype)
    image = image_create(NDIM, n, x, y, z)
    xdata[index] = image.reshape(NDIM, NDIM, NDIM, NCHANNEL)
    ydata[index] = RANDOM_SOURCE_LABEL

    metadata_arrays['target_atom_count'][index] = target_inner
    metadata_arrays['atom_count'][index] = actual_inner
    metadata_arrays['buffer_atom_count'][index] = n
    metadata_arrays['random_type'][index] = rtype
    metadata_arrays['ncluster'][index] = ncluster
    metadata_arrays['cluster_sigma'][index] = sigma
    metadata_arrays['random_attempts'][index] = attempts


def finish_metadata(seed_value, random_type_probs, metadata_arrays):
    metadata = {
        'seed': np.array(seed_value, dtype=np.int64),
        'lattice_name': np.array(LATTICE_NAME),
        'source_label': np.array(RANDOM_SOURCE_LABEL, dtype=np.int8),
        'source_label_name': np.array('random'),
        'source_label_names': np.array(['solid', 'random']),
        'buffer_margin': np.array(BUFFER_MARGIN, dtype=np.float32),
        'cluster_sigma_min': np.array(CLUSTER_SIGMA_MIN, dtype=np.float32),
        'cluster_sigma_max': np.array(CLUSTER_SIGMA_MAX, dtype=np.float32),
        'cluster_count_min': np.array(CLUSTER_COUNT_MIN, dtype=np.int32),
        'cluster_count_max': np.array(CLUSTER_COUNT_MAX, dtype=np.int32),
        'poisson_lambda_inner': np.array(POISSON_LAMBDA_INNER, dtype=np.float32),
        'random_type_alpha': RANDOM_TYPE_ALPHA,
        'random_type_probs': random_type_probs,
        'class_labels': np.array(['solid', 'random']),
        'random_type_labels': np.array([
            'gaussian_clustered',
            'poisson_inner_count_uniform',
            'poisson_count_gaussian_clustered',
        ]),
    }
    metadata.update(metadata_arrays)
    return metadata


def print_summary(ydata, random_type_probs, metadata_arrays, t0):
    atom_count = metadata_arrays['atom_count']
    buffer_atom_count = metadata_arrays['buffer_atom_count']
    random_type = metadata_arrays['random_type']
    cluster_sigma = metadata_arrays['cluster_sigma']
    print('total generation time:', time.time() - t0)
    print('source label counts:', np.bincount(np.asarray(ydata, dtype=np.int64), minlength=2))
    print('actual inner atom count min/mean/max:', atom_count.min(), atom_count.mean(), atom_count.max())
    print('buffer atom count min/mean/max:', buffer_atom_count.min(), buffer_atom_count.mean(), buffer_atom_count.max())
    print('sampled random type probabilities:', random_type_probs)
    print('random type counts:', np.bincount(random_type, minlength=len(random_type_probs)))
    print('cluster sigma min/mean/max:', cluster_sigma.min(), cluster_sigma.mean(), cluster_sigma.max())


def generate_dataset(nsample, seed_value):
    np.random.seed(seed_value)
    seed_numba_rng(seed_value)
    random_type_probs = np.random.dirichlet(RANDOM_TYPE_ALPHA)
    xdata = np.zeros((nsample, NDIM, NDIM, NDIM, NCHANNEL), dtype=np.float32)
    ydata = np.zeros(nsample, dtype=np.int8)
    metadata_arrays = make_metadata_arrays(nsample)
    t0 = time.time()
    t_print = t0

    for i in range(nsample):
        fill_random_sample(xdata, ydata, i, random_type_probs, metadata_arrays)
        if (i + 1) % NPRINT == 0:
            now = time.time()
            print('completed samples:', i + 1)
            print('time for last block:', now - t_print)
            t_print = now

    print_summary(ydata, random_type_probs, metadata_arrays, t0)
    return xdata, ydata, finish_metadata(seed_value, random_type_probs, metadata_arrays)


def write_dataset(nsample, seed_value, x_path, y_path, metadata_path):
    assert_outputs_writable([x_path, y_path, metadata_path])
    np.random.seed(seed_value)
    seed_numba_rng(seed_value)
    random_type_probs = np.random.dirichlet(RANDOM_TYPE_ALPHA)
    xdata = np.lib.format.open_memmap(
        x_path,
        mode='w+',
        dtype=np.float32,
        shape=(nsample, NDIM, NDIM, NDIM, NCHANNEL),
    )
    ydata = np.lib.format.open_memmap(y_path, mode='w+', dtype=np.int8, shape=(nsample,))
    metadata_arrays = make_metadata_arrays(nsample)
    t0 = time.time()
    t_print = t0

    for i in range(nsample):
        fill_random_sample(xdata, ydata, i, random_type_probs, metadata_arrays)
        if (i + 1) % NPRINT == 0:
            xdata.flush()
            ydata.flush()
            now = time.time()
            print('completed samples:', i + 1)
            print('time for last block:', now - t_print)
            t_print = now

    xdata.flush()
    ydata.flush()
    print_summary(ydata, random_type_probs, metadata_arrays, t0)
    np.savez(metadata_path, **finish_metadata(seed_value, random_type_probs, metadata_arrays))
