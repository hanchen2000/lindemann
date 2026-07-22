# Machine-Learned Structural Order and the Lindemann Criterion

This repository contains the data-generation scripts, vibration/RDF analysis
scripts, trained 3D CNN model, and training notebook used for the Lindemann
criterion study of BCC, FCC, SCC, and random/disordered structures.

The active combined classifier classes are:

```text
['bcc', 'fcc', 'scc', 'random']
```

HCP is not part of the active combined training set.

## Environment

Data generation requires a Python environment with:

```text
numpy
numba
```

Model training and model-based vibration testing require a TensorFlow/Keras
environment. Training is preferably run on a CUDA-capable GPU, because the
combined training data are large and the model is a 3D CNN.

Typical additional packages for the notebook and plotting are:

```text
jupyter
matplotlib
tensorflow
```

Use the TensorFlow installation method appropriate for your GPU, CUDA, and
driver setup.

## Generate BCC Training And Validation Data

From the repository root, run:

```bash
python bcc-traindata.py
python bcc-validationdata.py
```

The BCC training script generates 400,000 voxelized configurations and writes:

```text
xtrain-large-bcc400k-2.npy
ytrain-large-bcc400k-2.npy
metadata-train-large-bcc400k-2.npz
```

The BCC validation script generates 40,000 voxelized configurations and writes:

```text
xvalid-large-bcc40k.npy
yvalid-large-bcc40k.npy
metadata-valid-large-bcc40k.npz
```

The `.npy` voxel arrays are large. A single 400,000-sample training array is
about 52 GB, and a 40,000-sample validation array is about 5.2 GB. Make sure the
working directory has enough disk space before running the generators.

## Generate The Other Classes

The FCC, SCC, and random data are generated in the same way:

```bash
python fcc-traindata.py
python fcc-validationdata.py

python scc-traindata.py
python scc-validationdata.py

python random-traindata.py
python random-validationdata.py
```

The lattice generators use shared logic in `latticedata_common.py` and generate
only solid lattice samples. The random class is generated separately by
`randomdata_common.py` and the `random-*.py` scripts.

Each generator also writes metadata needed for reproducibility, including the
random seed, atom counts, lattice spacing, rotation parameters, and random-class
mixture information where applicable.

## Train The Combined Model

The combined 3D CNN training workflow is in:

```text
Phase-trainCombined.ipynb
```

Open the notebook in a TensorFlow/Keras Jupyter environment after generating the
training and validation arrays. The notebook trains the combined BCC/FCC/SCC/random
classifier and saves a Keras model.

This repository includes the trained model used for the paper:

```text
saved-model-combined-bcc-fcc-scc-random-best.keras
```

## Run Vibration And RDF Analyses

To evaluate the trained combined model under controlled vibrational disorder:

```bash
python combined-vibration-testing.py \
  --model saved-model-combined-bcc-fcc-scc-random-best.keras \
  --lattices bcc fcc scc
```

To compute RDF diagnostics for the same Lindemann-type displacement sweep:

```bash
python rdf-vibration-study.py --lattices bcc fcc scc
```

The model outputs can be interpreted as learned structural scores:

```text
P_bcc, P_fcc, P_scc, P_random
P_solid = P_bcc + P_fcc + P_scc
```

The vibration sweep also records the measured Lindemann ratio and
nearest-neighbor coefficient of variation.
