# %%
import genstudio.plot as Plot
import numpy as np
from genstudio.plot import js
from typing import Any
from colorsys import hsv_to_rgb


def make_torus_knot(n_points: int):
    # Create a torus knot
    t = np.linspace(0, 4 * np.pi, n_points)
    p, q = 3, 2  # Parameters that determine knot shape
    R, r = 2, 1  # Major and minor radii

    # Torus knot parametric equations
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = r * np.sin(q * t)

    # Add Gaussian noise to create volume
    noise_scale = 0.1
    x += np.random.normal(0, noise_scale, n_points)
    y += np.random.normal(0, noise_scale, n_points)
    z += np.random.normal(0, noise_scale, n_points)

    # Base color from position
    angle = np.arctan2(y, x)
    height = (z - z.min()) / (z.max() - z.min())
    radius = np.sqrt(x * x + y * y)
    radius_norm = (radius - radius.min()) / (radius.max() - radius.min())

    # Create hue that varies with angle and height
    hue = (angle / (2 * np.pi) + height) % 1.0
    # Saturation that varies with radius
    saturation = 0.8 + radius_norm * 0.2
    # Value/brightness
    value = 0.8 + np.random.uniform(0, 0.2, n_points)

    # Convert HSV to RGB
    colors = np.array([hsv_to_rgb(h, s, v) for h, s, v in zip(hue, saturation, value)])
    # Reshape colors to match xyz structure before flattening
    rgb = (colors.reshape(-1, 3) * 255).astype(np.uint8).flatten()

    # Prepare point cloud coordinates
    xyz = np.column_stack([x, y, z]).astype(np.float32).flatten()

    return xyz, rgb


def make_cube(n_points: int):
    # Create random points within a cube
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    z = np.random.uniform(-1, 1, n_points)

    # Create colors based on position in cube
    # Normalize positions to 0-1 range for colors
    x_norm = (x + 1) / 2
    y_norm = (y + 1) / 2
    z_norm = (z + 1) / 2

    # Create hue that varies with position
    hue = (x_norm + y_norm + z_norm) / 3
    # Saturation varies with distance from center
    distance = np.sqrt(x * x + y * y + z * z)
    distance_norm = distance / np.sqrt(3)  # normalize by max possible distance
    saturation = 0.7 + 0.3 * distance_norm
    # Value varies with height
    value = 0.7 + 0.3 * z_norm

    # Convert HSV to RGB
    colors = np.array([hsv_to_rgb(h, s, v) for h, s, v in zip(hue, saturation, value)])
    rgb = (colors.reshape(-1, 3) * 255).astype(np.uint8).flatten()

    # Prepare point cloud coordinates
    xyz = np.column_stack([x, y, z]).astype(np.float32).flatten()

    return xyz, rgb


# Create point clouds with 50k points
torus_xyz, torus_rgb = make_torus_knot(50000)
cube_xyz, cube_rgb = make_cube(50000)


class Points(Plot.LayoutItem):
    def __init__(self, props):
        self.props = props

    def for_json(self) -> Any:
        return [Plot.JSRef("points.PointCloudViewer"), self.props]


# Camera parameters - positioned to see the spiral structure
camera = {
    "position": [7, 4, 4],
    "target": [0, 0, 0],
    "up": [0, 0, 1],
    "fov": 40,
    "near": 0.1,
    "far": 2000,
}


def scene(controlled, point_size=4, xyz=torus_xyz, rgb=torus_rgb):
    cameraProps = (
        {"defaultCamera": camera}
        if not controlled
        else {
            "onCameraChange": js("(camera) => $state.update({'camera': camera})"),
            "camera": js("$state.camera"),
        }
    )
    return Points(
        {
            "points": {"xyz": xyz, "rgb": rgb},
            "backgroundColor": [0.1, 0.1, 0.1, 1],  # Dark background to make colors pop
            "className": "h-[400px] w-[400px]",
            "pointSize": point_size,
            "onPointClick": js(
                "(i) => $state.update({highlights: $state.highlights.includes(i) ? $state.highlights.filter(h => h !== i) : [...$state.highlights, i]})"
            ),
            "highlights": js("$state.highlights"),
            "onPointHover": js("(e) => null"),
            "highlightColor": [1.0, 1.0, 0.0],
            **cameraProps,
        }
    )


(
    Plot.initialState({"camera": camera, "highlights": []})
    | scene(True, 0.01, xyz=cube_xyz, rgb=cube_rgb) & scene(True, 1)
    | scene(False, 0.1, xyz=cube_xyz, rgb=cube_rgb) & scene(False, 0.5)
)
