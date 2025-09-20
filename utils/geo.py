def blend_weight(tile_size: int, overlap: int):
    import numpy as np
    s = tile_size
    ramp = np.hanning(2*overlap)
    core = np.ones(s-2*overlap)
    w1d = np.concatenate([ramp[:overlap], core, ramp[-overlap:]])
    win = np.outer(w1d, w1d)
    return win.astype('float32')
