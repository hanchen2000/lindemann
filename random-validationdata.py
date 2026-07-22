from randomdata_common import write_dataset

NVALID = 40000
SEED = 19283

if __name__ == '__main__':
    write_dataset(
        NVALID,
        SEED,
        'xvalid-large-random40k.npy',
        'yvalid-large-random40k.npy',
        'metadata-valid-large-random40k.npz',
    )
