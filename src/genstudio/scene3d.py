import genstudio.plot as Plot
from typing import Any, Dict, Union, Optional, TypedDict

import numpy as np

# Move Array type definition after imports
ArrayLike = Union[list, np.ndarray, Plot.JSExpr]
NumberLike = Union[int, float, np.number, Plot.JSExpr]


class Decoration(TypedDict, total=False):
    indexes: ArrayLike
    color: Optional[ArrayLike]  # [r,g,b]
    alpha: Optional[NumberLike]  # 0-1
    scale: Optional[NumberLike]  # scale factor
    minSize: Optional[NumberLike]  # pixels


def deco(
    indexes: Union[int, np.integer, ArrayLike],
    *,
    color: Optional[ArrayLike] = None,
    alpha: Optional[NumberLike] = None,
    scale: Optional[NumberLike] = None,
    min_size: Optional[NumberLike] = None,
) -> Decoration:
    """Create a decoration for scene components.

    Args:
        indexes: Single index or list of indices to decorate
        color: Optional RGB color override [r,g,b]
        alpha: Optional opacity value (0-1)
        scale: Optional scale factor
        min_size: Optional minimum size in pixels

    Returns:
        Dictionary containing decoration settings
    """
    # Convert single index to list
    if isinstance(indexes, (int, np.integer)):
        indexes = np.array([indexes])

    # Create base decoration dict with Any type to avoid type conflicts
    decoration: Dict[str, Any] = {"indexes": indexes}

    # Add optional parameters if provided
    if color is not None:
        decoration["color"] = color
    if alpha is not None:
        decoration["alpha"] = alpha
    if scale is not None:
        decoration["scale"] = scale
    if min_size is not None:
        decoration["minSize"] = min_size

    return decoration  # type: ignore


class SceneComponent(Plot.LayoutItem):
    """Base class for all 3D scene components."""

    def __init__(self, type_name: str, data: Dict[str, Any], **kwargs):
        super().__init__()
        self.type = type_name
        self.data = data
        self.decorations = kwargs.get("decorations")
        self.on_hover = kwargs.get("onHover")
        self.on_click = kwargs.get("onClick")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the element to a dictionary representation."""
        element = {"type": self.type, "data": self.data}
        if self.decorations:
            element["decorations"] = self.decorations
        if self.on_hover:
            element["onHover"] = self.on_hover
        if self.on_click:
            element["onClick"] = self.on_click
        return element

    def for_json(self) -> Dict[str, Any]:
        """Convert the element to a JSON-compatible dictionary."""
        return Scene(self).for_json()

    def __add__(
        self, other: Union["SceneComponent", "Scene", Dict[str, Any]]
    ) -> "Scene":
        """Allow combining components with + operator."""
        if isinstance(other, Scene):
            return other + self
        elif isinstance(other, SceneComponent):
            return Scene(self, other)
        elif isinstance(other, dict):
            return Scene(self, other)
        else:
            raise TypeError(f"Cannot add SceneComponent with {type(other)}")

    def __radd__(self, other: Dict[str, Any]) -> "Scene":
        """Allow combining components with + operator when dict is on the left."""
        return Scene(self, other)


class Scene(Plot.LayoutItem):
    """A 3D scene visualization component using WebGPU.

    This class creates an interactive 3D scene that can contain multiple types of components:
    - Point clouds
    - Ellipsoids
    - Ellipsoid bounds (wireframe)
    - Cuboids

    The visualization supports:
    - Orbit camera control (left mouse drag)
    - Pan camera control (shift + left mouse drag or middle mouse drag)
    - Zoom control (mouse wheel)
    - Component hover highlighting
    - Component click selection
    """

    def __init__(
        self,
        *components_and_props: Union[SceneComponent, Dict[str, Any]],
    ):
        """Initialize the scene.

        Args:
            *components_and_props: Scene components and optional properties.
        """
        components = []
        scene_props = {}
        for item in components_and_props:
            if isinstance(item, SceneComponent):
                components.append(item)
            elif isinstance(item, dict):
                scene_props.update(item)
            else:
                raise TypeError(f"Invalid type in components_and_props: {type(item)}")

        self.components = components
        self.scene_props = scene_props
        super().__init__()

    def __add__(self, other: Union[SceneComponent, "Scene", Dict[str, Any]]) -> "Scene":
        """Allow combining scenes with + operator."""
        if isinstance(other, Scene):
            return Scene(*self.components, *other.components, self.scene_props)
        elif isinstance(other, SceneComponent):
            return Scene(*self.components, other, self.scene_props)
        elif isinstance(other, dict):
            return Scene(*self.components, {**self.scene_props, **other})
        else:
            raise TypeError(f"Cannot add Scene with {type(other)}")

    def __radd__(self, other: Dict[str, Any]) -> "Scene":
        """Allow combining scenes with + operator when dict is on the left."""
        return Scene(*self.components, {**other, **self.scene_props})

    def for_json(self) -> Any:
        """Convert to JSON representation for JavaScript."""
        components = [
            e.to_dict() if isinstance(e, SceneComponent) else e for e in self.components
        ]

        props = {"components": components, **self.scene_props}

        return [Plot.JSRef("scene3d.Scene"), props]


def flatten_array(
    arr: ArrayLike, dtype: Any = np.float32
) -> Union[np.ndarray, Plot.JSExpr]:
    """Flatten an array if it is a 2D array, otherwise return as is.

    Args:
        arr: The array to flatten.
        dtype: The desired data type of the array.

    Returns:
        A flattened array if input is 2D, otherwise the original array.
    """
    if isinstance(arr, (np.ndarray, list)):
        arr = np.asarray(arr, dtype=dtype)
        if arr.ndim == 2:
            return arr.flatten()
    return arr


def PointCloud(
    positions: ArrayLike,
    colors: Optional[ArrayLike] = None,
    scales: Optional[ArrayLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create a point cloud element.

    Args:
        positions: Nx3 array of point positions or flattened array
        colors: Nx3 array of RGB colors or flattened array (optional)
        scales: N array of point scales or flattened array (optional)
        **kwargs: Additional arguments like decorations, onHover, onClick
    """
    positions = flatten_array(positions, dtype=np.float32)
    data: Dict[str, Any] = {"positions": positions}

    if colors is not None:
        data["colors"] = flatten_array(colors, dtype=np.float32)

    if scales is not None:
        data["scales"] = flatten_array(scales, dtype=np.float32)

    return SceneComponent("PointCloud", data, **kwargs)


def Ellipsoid(
    centers: ArrayLike,
    radii: ArrayLike,
    colors: Optional[ArrayLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create an ellipsoid element.

    Args:
        centers: Nx3 array of ellipsoid centers or flattened array
        radii: Nx3 array of radii (x,y,z) or flattened array
        colors: Nx3 array of RGB colors or flattened array (optional)
        **kwargs: Additional arguments like decorations, onHover, onClick
    """
    centers = flatten_array(centers, dtype=np.float32)
    radii = flatten_array(radii, dtype=np.float32)
    data = {"centers": centers, "radii": radii}

    if colors is not None:
        data["colors"] = flatten_array(colors, dtype=np.float32)

    return SceneComponent("Ellipsoid", data, **kwargs)


def EllipsoidAxes(
    centers: ArrayLike,
    radii: ArrayLike,
    colors: Optional[ArrayLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create an ellipsoid bounds (wireframe) element.

    Args:
        centers: Nx3 array of ellipsoid centers or flattened array
        radii: Nx3 array of radii (x,y,z) or flattened array
        colors: Nx3 array of RGB colors or flattened array (optional)
        **kwargs: Additional arguments like decorations, onHover, onClick
    """
    centers = flatten_array(centers, dtype=np.float32)
    radii = flatten_array(radii, dtype=np.float32)
    data = {"centers": centers, "radii": radii}

    if colors is not None:
        data["colors"] = flatten_array(colors, dtype=np.float32)

    return SceneComponent("EllipsoidAxes", data, **kwargs)


def Cuboid(
    centers: ArrayLike,
    sizes: ArrayLike,
    colors: Optional[ArrayLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create a cuboid element.

    Args:
        centers: Nx3 array of cuboid centers or flattened array
        sizes: Nx3 array of sizes (width,height,depth) or flattened array
        colors: Nx3 array of RGB colors or flattened array (optional)
        **kwargs: Additional arguments like decorations, onHover, onClick
    """
    centers = flatten_array(centers, dtype=np.float32)
    sizes = flatten_array(sizes, dtype=np.float32)
    data = {"centers": centers, "sizes": sizes}

    if colors is not None:
        data["colors"] = flatten_array(colors, dtype=np.float32)

    return SceneComponent("Cuboid", data, **kwargs)
