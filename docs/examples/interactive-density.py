# %% [markdown]
# # Interactive Density Plot
#
# This example demonstrates how to create an interactive density plot that generates random points following a normal distribution around clicked locations. It combines several key features:
#
# - State management with `Plot.initialState`
# - Event handling with `Plot.events`
# - Multiple layered marks (density plot and scatter plot)
# - JavaScript-based point generation using Box-Muller transform
#
# Click anywhere on the plot to generate a cluster of normally distributed points:

# %%

import genstudio.plot as Plot
from genstudio.plot import js

# Create a scatter plot with interactive point generation
(
    Plot.initialState({"points": []}, sync=True)
    | (
        Plot.density(js("$state.points"))
        + Plot.dot(js("$state.points"))
        + Plot.events(
            {
                "onClick": js("""(e) => {
                const std = 0.05;
                const points = Array.from({length: 20}, () => {
                    const r = Math.sqrt(-2 * Math.log(Math.random()));
                    const theta = 2 * Math.PI * Math.random();
                    return [
                        e.x + std * r * Math.cos(theta),
                        e.y + std * r * Math.sin(theta)
                    ];
                });
                $state.update(['points', 'concat', points]);
            }""")
            }
        )
        + Plot.domain([0, 1])
    )
    | [
        "button",
        {
            "onClick": lambda widget, _: print(widget.state.points),
            "className": "bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded",
        },
        "Print Points",
    ]
)
