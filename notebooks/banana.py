from genstudio.scene3d import Ellipsoid
import pickle
import genstudio.plot as Plot

with open("./notebooks/banana_gaussians.pkl", "rb") as f:
    banana_gaussians = pickle.load(f)


def render_gaussians(bananas):
    """
    Renders a Gen3D state's Gaussian ellipsoids using genstudio.scene3d.

    Parameters:
        state: A Gen3D state object containing Gaussian parameters

    Returns:
        A genstudio Scene3D containing the rendered ellipsoids
    """

    # Convert covariances to ellipsoid parameters using gen3d's function

    return (
        Ellipsoid(
            centers=bananas["xyz"],
            half_sizes=bananas["half_sizes"] * 2.5,
            quaternions=bananas["quaternions"],
            colors=bananas["colors"],
            fill_mode="MajorWireframe",
        )
        + {
            "camera": Plot.ref(
                {
                    "position": [0.045741, 0.137745, 0.362853],
                    "target": [0.000000, 0.000000, 0.000000],
                    "up": [0.000000, 1.000000, 0.000000],
                    "fov": 45,
                },
                "camera",
            ),
            "onCameraChange": Plot.js("(camera) => $state.update({camera})"),
        }
    ) | Plot.Slider("i", range=len(bananas["xyz"] / 3))
    #


render_gaussians(banana_gaussians[0])
