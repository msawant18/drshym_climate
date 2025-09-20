# Model Card — DrShym UNet (ResNet18 encoder)

**Intended use**: Binary flood extent segmentation on Sentinel‑1 SAR VV IW GRD, 10 m.

**Training data**: Weak labels from water occurrence maps + elevation masks, corrected subset of ~100 tiles for clean validation.

**Geography/time**: Historical flood event (fill when chosen).

**Augmentation**: flips, small rotations, intensity jitter.

**Hyperparameters**: see `configs/flood.yaml`.

**Calibration**: Temperature scaling on validation split; ECE reduced ≥30% target.

**Limitations**: Look‑alikes (asphalt, layover/shadow) and steep slopes can cause FP. Do not claim absolute flooded area without uncertainty.

**Ethics**: Avoid overclaiming; provide confidence; keep CRS and provenance.
