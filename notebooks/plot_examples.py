# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
%load_ext autoreload
%autoreload 2

import genstudio.plot as Plot
import numpy as np
import genjax as genjax
from genjax import gen
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np


# %% [markdown]
# ## Approach
#
# - The [pyobsplot](https://github.com/juba/pyobsplot) library creates "stubs" in python which directly mirror the Observable Plot API. An AST-like "spec" is created in python and then interpreted in javascript.
# - The [Observable Plot](https://observablehq.com/plot/) library does not have "chart types" but rather "marks", which are layered to produce a chart. These are composable via `+` in Python.
#
# ## Instructions
#
# The starting point for seeing what's possible is the [Observable Plot](https://observablehq.com/plot/what-is-plot) website.
# Plots are composed of **marks**, and you'll want to familiarize yourself with the available marks and how they're created.
#
# Generate random data from a normal distribution
# %%
def normal_100():
    return np.random.normal(loc=0, scale=1, size=1000)


# %% [markdown]
# #### Histogram
# %%
Plot.histogram(normal_100())

# %% [markdown]
# #### Scatter and Line plots
# Unlike other mark types which expect a single values argument, `dot` and `line`
# also accept separate `xs` and `ys` for passing in columnar data (usually the case
# when working with jax.)
# %%
Plot.dot({"x": normal_100(), "y": normal_100()}) + Plot.frame()

# %% [markdown]
# #### One-dimensional heatmap
# %%

(
    Plot.rect(normal_100(), Plot.binX({"fill": "count"}))
    + Plot.color_scheme("YlGnBu")
    + {"height": 75}
)

# %% [markdown]
# #### Plot.doc
# 
# Plot.doc(Plot.foo) will render a markdown-formatted docstring when available:
# %%

Plot.doc(Plot.line)

# %% [markdown]
# #### Plot composition
#
# Marks and options can be composed by including them as arguments to `Plot.new(...)`,
# or by adding them to a plot. Adding marks or options does not change the underlying plot,
# so you can re-use plots in different combinations.

# %%
circle = Plot.dot([[0, 0]], r=100)
circle

# %%
circle + Plot.frame() + {"inset": 50}

# %% [markdown]
#
# A GenJAX example

# A regression distribution.
# %%

key = jrand.PRNGKey(314159)

@gen
def regression(x, coefficients, sigma):
    basis_value = jnp.array([1.0, x, x**2])
    polynomial_value = jnp.sum(basis_value * coefficients)
    y = genjax.normal(polynomial_value, sigma) @ "v"
    return y


# %% [markdown]
# Regression, with an outlier random variable.
# %%

@gen
def regression_with_outlier(x, coefficients):
    is_outlier = genjax.flip(0.1) @ "is_outlier"
    sigma = jnp.where(is_outlier, 30.0, 0.3)
    is_outlier = jnp.array(is_outlier, dtype=int)
    return regression(x, coefficients, sigma) @ "y"


# The full model, sample coefficients for a curve, and then use
# them in independent draws from the regression submodel.
@gen
def full_model(xs):
    coefficients = (
        genjax.mv_normal(
            jnp.zeros(3, dtype=float),
            2.0 * jnp.identity(3),
        )
        @ "alpha"
    )
    ys = regression_with_outlier.vmap(in_axes=(0, None))(xs, coefficients) @ "ys"
    return ys


data = jnp.arange(0, 10, 0.5)
key, sub_key = jrand.split(key)
tr = jax.jit(full_model.simulate)(sub_key, (data,))

key, *sub_keys = jrand.split(key, 10)
traces = jax.vmap(lambda k: full_model.simulate(k, (data,)))(jnp.array(sub_keys))

traces

# %% [markdown]
# #### Multi-dimensional (nested) data
#
# Data from GenJAX often comes in the form of multi-dimensional (nested) lists.
# To prepare data for plotting, we can describe these dimensions using `Plot.dimensions`.
# %%

ys = traces.get_choices()["ys", ..., "y", "v"]
data = Plot.dimensions(ys, ["sample", "ys"], leaves="y")

# => <Dimensioned shape=(9, 20), names=['sample', 'ys', 'y']>

data.flatten()
# => [{'sample': 0, 'ys': 0, 'y': Array(0.11651635, dtype=float32)},
#     {'sample': 0, 'ys': 1, 'y': Array(-5.046837, dtype=float32)},
#     {'sample': 0, 'ys': 2, 'y': Array(-0.9120707, dtype=float32)},
#     {'sample': 0, 'ys': 3, 'y': Array(0.4919241, dtype=float32)},
#     {'sample': 0, 'ys': 4, 'y': Array(1.081743, dtype=float32)},
#     {'sample': 0, 'ys': 5, 'y': Array(1.6471565, dtype=float32)},
#     {'sample': 0, 'ys': 6, 'y': Array(3.6472352, dtype=float32)},
#     {'sample': 0, 'ys': 7, 'y': Array(5.080149, dtype=float32)},
#     {'sample': 0, 'ys': 8, 'y': Array(6.961242, dtype=float32)},
#     {'sample': 0, 'ys': 9, 'y': Array(10.374397, dtype=float32)} ...]

#%%

# %% [markdown]
#
# When passed to a plotting function, this annotated dimensional data will be flattened into
# a single list of objects, with entries for each dimension and leaf name. Here, we'll call
# .flatten() directly in python, but in practice the arrays will be flattened after (de)serialization
# to our JavaScript rendering environment.
# %%

Plot.dimensions(ys, ["sample", "ys"], leaves="y").flatten()[:10]

# %% [markdown]
# #### Small Multiples
#
# The `facetGrid` option splits a dataset by key, and shows each split in its own chart
# with consistent scales.
# %%
(
    Plot.dot(
        Plot.dimensions(ys, ["sample", "ys"], leaves="y"),
        facetGrid="sample",
        x=Plot.repeat(data),
        y="y",
    )
    + {"height": 600}
    + Plot.frame()
)

# %% [markdown]
# `Plot.get_in` reads data from a nested structure, giving names to dimensions and leaves
# along the way in a single step. It works with Python lists/dicts as well as GenJAX
# traces and choicemaps. Here we'll construct a synthetic dataset and plot using `get_in`.
# %%

import random

bean_data = [[0 for _ in range(8)]]
for day in range(1, 21):
    rained = random.choices([True, False], weights=[1, 3])[
        0
    ]  # Decide if it rained with a 1 in 4 chance
    precipitation = 0 if not random.choice([True, False]) else random.uniform(0.1, 3)
    min_growth = 0.1 + (precipitation * 0.5)
    max_growth = 1 + (precipitation * 5)
    bean_data.append(
        [height + random.uniform(min_growth, max_growth) for height in bean_data[-1]]
    )
bean_data

# %% [markdown]
# Using `get_in` we've given names to each level of nesting (and leaf values), which we can see in the metadata
# of the Dimensioned object:
# %%

data = Plot.get_in(bean_data, [{...: "day"}, {...: "bean"}, {"leaves": "height"}])
# => <Dimensioned shape=(21, 8), names=['day', 'bean', 'height']>

data.flatten()
# => [{'day': 0, 'bean': 0, 'height': 0},
#     {'day': 0, 'bean': 1, 'height': 0},
#     {'day': 0, 'bean': 2, 'height': 0},
#     {'day': 0, 'bean': 3, 'height': 0},
#     {'day': 0, 'bean': 4, 'height': 0},
#     {'day': 0, 'bean': 5, 'height': 0},
#     {'day': 0, 'bean': 6, 'height': 0},
#     {'day': 0, 'bean': 7, 'height': 0},
#     {'day': 1, 'bean': 0, 'height': 0.17486922945122418},
#     {'day': 1, 'bean': 1, 'height': 0.8780341204172442},
#     {'day': 1, 'bean': 2, 'height': 0.6476780304516665},
#     {'day': 1, 'bean': 3, 'height': 0.9339147036777222}, ...]

# %%[markdown]
# Now that our dimensions and leaf have names, we can pass them as options to `Plot.dot`.
# Here we'll use the `facetGrid` option to render a separate plot for each bean.
# %%
Plot.dot(data, {"x": "day", "y": "height", "facetGrid": "bean"}) + Plot.frame()

# %% [markdown]
# Let's draw a line for each bean to plot its growth over time. The `z` channel splits the data into
# separate lines.
# %%
(
    Plot.line(
        data, {"x": "day", "y": "height", "z": "bean", "stroke": "bean"}
    )
    + Plot.frame()
)

#%%
Plot.View.domainTest(Plot.dimensions(bean_data, ["day", "bean"], leaves="height"))