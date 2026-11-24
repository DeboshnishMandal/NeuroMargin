import numpy as np
from src.neuromargin.audio import extract_features, SR

def test_feature_vector_is_finite_and_stable():
    t=np.arange(SR*3)/SR
    y=(0.2*np.sin(2*np.pi*180*t)).astype("float32")
    f=extract_features(y)
    assert len(f)>50
    assert all(np.isfinite(list(f.values())))
