import os
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

NDIM = 32
NCHANNEL = 1
NMAX = 5000
NPRINT = 1000
TARGET_ATOM_COUNT_MIN = 25
TARGET_ATOM_COUNT_MAX = 60
MAX_LATTICE_ATTEMPTS = 200
BUFFER_MARGIN = np.float32(0.35)
SOLID_SOURCE_LABEL = np.int8(0)


@njit
def seed_numba_rng(seed_value):
    np.random.seed(seed_value)


@njit
def random_unit_quaternion():
    u1 = np.random.rand()
    u2 = np.random.rand()
    u3 = np.random.rand()
    r1 = np.sqrt(1.0 - u1)
    r2 = np.sqrt(u1)
    t1 = 2.0 * np.pi * u2
    t2 = 2.0 * np.pi * u3
    x = r1 * np.sin(t1)
    y = r1 * np.cos(t1)
    z = r2 * np.sin(t2)
    w = r2 * np.cos(t2)
    return np.float32(w), np.float32(x), np.float32(y), np.float32(z)


@njit
def quaternion_rotation_matrix(w, x, y, z):
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float32)


@njit
def true_lattice_with_metadata(nmax, cell_atoms, basis_fraction):
    x_all = np.zeros(nmax, dtype=np.float32)
    y_all = np.zeros(nmax, dtype=np.float32)
    z_all = np.zeros(nmax, dtype=np.float32)
    x_inner = np.zeros(nmax, dtype=np.float32)
    y_inner = np.zeros(nmax, dtype=np.float32)
    z_inner = np.zeros(nmax, dtype=np.float32)
    n_all = 0
    n_inner = 0

    target_count = np.random.randint(TARGET_ATOM_COUNT_MIN, TARGET_ATOM_COUNT_MAX + 1)
    a = np.float32((cell_atoms / target_count) ** (1.0 / 3.0))

    qw, qx, qy, qz = random_unit_quaternion()
    r = quaternion_rotation_matrix(qw, qx, qy, qz)

    v1 = np.array([a, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, a, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, a], dtype=np.float32)
    rv1 = r @ v1
    rv2 = r @ v2
    rv3 = r @ v3

    nbasis = basis_fraction.shape[0]
    basis = np.zeros((nbasis, 3), dtype=np.float32)
    for ib in range(nbasis):
        unrotated = np.array([
            basis_fraction[ib, 0] * a,
            basis_fraction[ib, 1] * a,
            basis_fraction[ib, 2] * a,
        ], dtype=np.float32)
        basis[ib] = r @ unrotated

    origin = np.array([np.random.rand(), np.random.rand(), np.random.rand()], dtype=np.float32)
    min_len = min(np.sqrt((rv1 * rv1).sum()), np.sqrt((rv2 * rv2).sum()), np.sqrt((rv3 * rv3).sum()))
    maxr = int(np.ceil((2.4 + 2.0 * BUFFER_MARGIN) / min_len)) + 2

    for i in range(-maxr, maxr + 1):
        for j in range(-maxr, maxr + 1):
            for k in range(-maxr, maxr + 1):
                base = origin + i * rv1 + j * rv2 + k * rv3
                for ib in range(nbasis):
                    p = base + basis[ib]
                    x = p[0]
                    y = p[1]
                    z = p[2]
                    if (-BUFFER_MARGIN <= x <= 1.0 + BUFFER_MARGIN and
                        -BUFFER_MARGIN <= y <= 1.0 + BUFFER_MARGIN and
                        -BUFFER_MARGIN <= z <= 1.0 + BUFFER_MARGIN):
                        if n_all < nmax:
                            x_all[n_all] = x
                            y_all[n_all] = y
                            z_all[n_all] = z
                            n_all += 1
                    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= z <= 1.0:
                        if n_inner < nmax:
                            x_inner[n_inner] = x
                            y_inner[n_inner] = y
                            z_inner[n_inner] = z
                            n_inner += 1

    scaling = int(np.ceil(1.0 / a))
    return x_all, y_all, z_all, n_all, x_inner, y_inner, z_inner, n_inner, a, qw, qx, qy, qz, scaling


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


def sample_lattice_positions(lattice_name, cell_atoms, basis_fraction):
    for attempt in range(MAX_LATTICE_ATTEMPTS):
        out = true_lattice_with_metadata(NMAX, cell_atoms, basis_fraction)
        n_inner = out[7]
        if TARGET_ATOM_COUNT_MIN <= n_inner <= TARGET_ATOM_COUNT_MAX:
            return out, attempt + 1
    raise RuntimeError(f'Could not generate {lattice_name} sample in target atom-count range')


def fill_lattice_sample(xdata, ydata, index, lattice_name, cell_atoms, basis_fraction, metadata_arrays):
    out, attempts = sample_lattice_positions(lattice_name, cell_atoms, basis_fraction)
    xbuf, ybuf, zbuf, nbuf, xin, yin, zin, n_inner, a, qw, qx, qy, qz, scaling = out
    image = image_create(NDIM, nbuf, xbuf, ybuf, zbuf)
    xdata[index] = image.reshape(NDIM, NDIM, NDIM, NCHANNEL)
    ydata[index] = SOLID_SOURCE_LABEL

    metadata_arrays['atom_count'][index] = n_inner
    metadata_arrays['buffer_atom_count'][index] = nbuf
    metadata_arrays['lattice_a'][index] = a
    metadata_arrays['rotation_quaternion'][index] = (qw, qx, qy, qz)
    metadata_arrays['scaling'][index] = scaling
    metadata_arrays['lattice_attempts'][index] = attempts


def make_metadata_arrays(nsample):
    return {
        'atom_count': np.zeros(nsample, dtype=np.int32),
        'buffer_atom_count': np.zeros(nsample, dtype=np.int32),
        'lattice_a': np.zeros(nsample, dtype=np.float32),
        'rotation_quaternion': np.zeros((nsample, 4), dtype=np.float32),
        'scaling': np.zeros(nsample, dtype=np.int32),
        'lattice_attempts': np.zeros(nsample, dtype=np.int32),
    }


def finish_metadata(seed_value, lattice_name, cell_atoms, basis_fraction, metadata_arrays):
    metadata = {
        'seed': np.array(seed_value, dtype=np.int64),
        'lattice_name': np.array(lattice_name),
        'source_label': np.array(SOLID_SOURCE_LABEL, dtype=np.int8),
        'source_label_name': np.array('solid'),
        'source_label_names': np.array(['solid', 'random']),
        'cell_atoms': np.array(cell_atoms, dtype=np.float32),
        'basis_fraction': basis_fraction.astype(np.float32),
        'buffer_margin': np.array(BUFFER_MARGIN, dtype=np.float32),
        'rotation_parameterization': np.array('uniform_quaternion_wxyz'),
    }
    metadata.update(metadata_arrays)
    return metadata


def print_summary(ydata, metadata_arrays, t0):
    atom_count = metadata_arrays['atom_count']
    buffer_atom_count = metadata_arrays['buffer_atom_count']
    print('total generation time:', time.time() - t0)
    print('source label counts:', np.bincount(np.asarray(ydata, dtype=np.int64), minlength=2))
    print('inner atom count min/mean/max:', atom_count.min(), atom_count.mean(), atom_count.max())
    print('buffer atom count min/mean/max:', buffer_atom_count.min(), buffer_atom_count.mean(), buffer_atom_count.max())


def generate_lattice_dataset(nsample, seed_value, lattice_name, cell_atoms, basis_fraction):
    np.random.seed(seed_value)
    seed_numba_rng(seed_value)
    xdata = np.zeros((nsample, NDIM, NDIM, NDIM, NCHANNEL), dtype=np.float32)
    ydata = np.zeros(nsample, dtype=np.int8)
    metadata_arrays = make_metadata_arrays(nsample)
    t0 = time.time()
    t_print = t0

    for i in range(nsample):
        fill_lattice_sample(xdata, ydata, i, lattice_name, cell_atoms, basis_fraction, metadata_arrays)
        if (i + 1) % NPRINT == 0:
            now = time.time()
            print('completed samples:', i + 1)
            print('time for last block:', now - t_print)
            t_print = now

    print_summary(ydata, metadata_arrays, t0)
    return xdata, ydata, finish_metadata(seed_value, lattice_name, cell_atoms, basis_fraction, metadata_arrays)


def write_lattice_dataset(nsample, seed_value, lattice_name, cell_atoms, basis_fraction, x_path, y_path, metadata_path):
    assert_outputs_writable([x_path, y_path, metadata_path])
    np.random.seed(seed_value)
    seed_numba_rng(seed_value)
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
        fill_lattice_sample(xdata, ydata, i, lattice_name, cell_atoms, basis_fraction, metadata_arrays)
        if (i + 1) % NPRINT == 0:
            xdata.flush()
            ydata.flush()
            now = time.time()
            print('completed samples:', i + 1)
            print('time for last block:', now - t_print)
            t_print = now

    xdata.flush()
    ydata.flush()
    print_summary(ydata, metadata_arrays, t0)
    np.savez(metadata_path, **finish_metadata(seed_value, lattice_name, cell_atoms, basis_fraction, metadata_arrays))


SCC_BASIS = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
BCC_BASIS = np.array([[0.0, 0.0, 0.0],
                      [0.5, 0.5, 0.5]], dtype=np.float32)
FCC_BASIS = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5],
                      [0.5, 0.5, 0.0]], dtype=np.float32)
