import numpy as np

from latticedata_common import (
    FCC_BASIS,
    NMAX,
    generate_lattice_dataset,
    image_create,
    seed_numba_rng,
    true_lattice_with_metadata as _true_lattice_with_metadata,
    write_lattice_dataset,
)

LATTICE_NAME = 'FCC'
CELL_ATOMS = np.float32(4.0)
BASIS_FRACTION = FCC_BASIS
NTRAIN = 400000
SEED = 79481


def true_lattice_with_metadata(nmax):
    return _true_lattice_with_metadata(nmax, CELL_ATOMS, BASIS_FRACTION)


def generate_dataset(nsample, seed_value):
    return generate_lattice_dataset(nsample, seed_value, LATTICE_NAME, CELL_ATOMS, BASIS_FRACTION)


if __name__ == '__main__':
    write_lattice_dataset(
        NTRAIN,
        SEED,
        LATTICE_NAME,
        CELL_ATOMS,
        BASIS_FRACTION,
        'xtrain-large-fcc400k-2.npy',
        'ytrain-large-fcc400k-2.npy',
        'metadata-train-large-fcc400k-2.npz',
    )
