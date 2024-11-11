# %% [markdown]
# # Animated Pixel Display
#
# This example demonstrates how to create an animated pixel display using NumPy arrays and `genstudio.plot`.
#
#
# First, let's import the required libraries:

# %%
import genstudio.plot as Plot
import numpy as np
from genstudio.plot import js

# %% [markdown]
# ## Generate Pixels
#
# The `generate_pixels` function creates animated RGB frames using phase-shifted circular waves, returning a list of frames, each of which is a flattened uint8 array.


# %%
def generate_pixels(width=100, height=100, num_frames=60):
    # Generate animated RGB frames with phase-shifted circular waves
    x, y = np.meshgrid(np.linspace(-4, 4, width), np.linspace(-4, 4, height))
    t = np.linspace(0, 2 * np.pi, num_frames)[:, None, None]
    r = np.sqrt(x**2 + y**2)

    intensity = np.sin(r - t) * 255
    rgb = np.stack(
        [
            np.clip(intensity * np.sin(t + phase), 0, 255)
            for phase in [0, 2 * np.pi / 3, 4 * np.pi / 3]
        ],
        axis=-1,
    )

    return list(rgb.reshape(num_frames, -1).astype(np.uint8))


# %% [markdown]
# ## Plot a Single Frame
#
# Let's start by plotting a single frame of our pixel data. We'll use `Plot.pixels` to display the RGB values:

# %%
# Generate a single frame
single_frame = generate_pixels(width=50, height=50, num_frames=1)[0]

# Plot it using Plot.pixels
Plot.pixels(single_frame, imageWidth=50, imageHeight=50)


# %% [markdown]
# ## Create Interactive Display
#
# Now let's set up the visualization with animation controls. We manage the visualization state using `Plot.initialState`, which creates a shared state object accessible from both Python and JavaScript. The state contains our pixel data array, dimensions, current frame index, and animation speed. The Plot.pixels component reads the current frame from state using js("$state.pixels[$state.frame]"), while the slider component updates the frame index in state as it moves. This state-driven approach allows smooth coordination between the display and controls.

# %%
width = 50
height = 50
num_frames = 10
fps = 30

data = generate_pixels(width=width, height=height, num_frames=num_frames)

(
    Plot.initialState(
        {"pixels": data, "width": width, "height": height, "frame": 0, "fps": fps}
    )
    | Plot.pixels(
        js("$state.pixels[$state.frame]"),
        imageWidth=js("$state.width"),
        imageHeight=js("$state.height"),
    )
    | Plot.Slider(
        "frame", rangeFrom=js("$state.pixels"), showFps=True, fps=js("$state.fps")
    )
)
