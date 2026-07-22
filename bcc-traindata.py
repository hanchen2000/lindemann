import numpy as np

from latticedata_common import (
    BCC_BASIS,
    NMAX,
    generate_lattice_dataset,
    image_create,
    seed_numba_rng,
    true_lattice_with_metadata as _true_lattice_with_metadata,
    write_lattice_dataset,
)

LATTICE_NAME = 'BCC'
CELL_ATOMS = np.float32(2.0)
BASIS_FRACTION = BCC_BASIS
NTRAIN = 400000
SEED = 89881


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
        'xtrain-large-bcc400k-2.npy',
        'ytrain-large-bcc400k-2.npy',
        'metadata-train-large-bcc400k-2.npz',
    )
