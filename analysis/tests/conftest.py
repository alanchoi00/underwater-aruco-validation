import numpy as np
import pytest

# The real ZED rectified intrinsics. No refraction correction: the pilot bundle
# measured fy/fx = 1.00 and f = 797 +/- 10%, rejecting the flat-port 1.333x model.
K_TRUE = dict(fx=797.54, fy=797.54, cx=483.78, cy=280.30)

# Layout recovered from the pilot bundle, RESCALED by the 0.9589 print scale (the
# bundle solved it under the nominal sizes, so it came out 4.1% too large).
# Metres, board frame anchored at 201. 202 tx = 0.4268 -> 420 mm of A4 page centres
# plus ~7 mm of tape gap, which is what the frames show.
LAYOUT_TRUE = np.array([
    [0.0000,  0.0000, 0.0],           # 201 (gauge)
    [0.4268,  0.0001, 0.0122],        # 202
    [0.1698,  0.0501, -0.0021],       # 301
    [0.2558,  0.0692, 0.0014],        # 302
    [0.1504, -0.0441, 0.0023],        # 303
    [0.2179, -0.0443, 0.0021],        # 304
    [0.1504, -0.1199, 0.0033],        # 305
    [0.2121, -0.1138, 0.0051],        # 401
    [0.2675, -0.1136, 0.0070],        # 402
])


@pytest.fixture
def K():
    return dict(K_TRUE)


@pytest.fixture
def layout_true():
    return LAYOUT_TRUE.copy()


@pytest.fixture
def synth_views():
    """Deterministic camera poses viewing the board at varied range and tilt.

    Tilt diversity is what makes focal length observable (Zhang). A fronto-parallel
    only set would leave f unidentifiable, so vary yaw/pitch deliberately.
    """
    rng = np.random.default_rng(0)
    poses = []
    for z in (0.8, 1.2, 1.8, 2.6, 3.5):
        for yaw in (-0.5, -0.25, 0.0, 0.25, 0.5):
            rv = np.array([rng.uniform(-0.15, 0.15), yaw, rng.uniform(-0.1, 0.1)])
            tv = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1), z])
            poses.append((rv, tv))
    return poses
