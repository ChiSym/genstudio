import uuid
from typing import Any, Sequence

from html2image import Html2Image
from PIL import Image

from genstudio.js_modules import JSRef
from genstudio.util import PARENT_PATH, CONFIG
from genstudio.widget import Widget, to_json

View = JSRef("View")


def html_snippet(data, id=None):
    id = id or f"genstudio-widget-{uuid.uuid4().hex}"
    serialized_data = to_json(data)

    # Read and inline the JS and CSS files
    with open(PARENT_PATH / "js/widget_build.js", "r") as js_file:
        js_content = js_file.read()
    with open(PARENT_PATH / "widget.css", "r") as css_file:
        css_content = css_file.read()

    html_content = f"""
    <style>{css_content}</style>
    <div id="{id}"></div>
    <script type="module">
        {js_content}
        const container = document.getElementById('{id}');
        const data = {serialized_data};
        renderData(container, data);
    </script>
    """

    return html_content


def html_standalone(data, id=None):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>GenStudio Widget</title>
    </head>
    <body>
        {html_snippet(data, id)}
    </body>
    </html>
    """


class HTML:
    def __init__(self, data):
        self.data = data
        self.id = f"genstudio-widget-{uuid.uuid4().hex}"

    def _repr_mimebundle_(self, **kwargs):
        html_content = html_snippet(self.data, self.id)
        return {"text/html": html_content}, {}


class LayoutItem:
    def __init__(self):
        self._html: HTML | None = None
        self._widget: Widget | None = None
        self._display_as = None

    def display_as(self, display_as) -> "LayoutItem":
        if display_as not in ["html", "widget"]:
            raise ValueError("display_pref must be either 'html' or 'widget'")
        self._display_as = display_as
        return self

    def to_json(self) -> dict[str, Any]:
        raise NotImplementedError("Subclasses must implement to_json method")

    def __and__(self, other: "LayoutItem") -> "Row":
        return Row(self, other)

    def __rand__(self, other: "LayoutItem") -> "Row":
        return Row(other, self)

    def __or__(self, other: "LayoutItem") -> "Column":
        return Column(self, other)

    def __ror__(self, other: "LayoutItem") -> "Column":
        return Column(other, self)

    def _repr_mimebundle_(self, **kwargs: Any) -> Any:
        display_as = self._display_as or CONFIG["display_as"]
        if display_as == "widget":
            return self.widget()._repr_mimebundle_(**kwargs)
        else:
            return self.html()._repr_mimebundle_(**kwargs)

    def html(self) -> HTML:
        """
        Lazily generate & cache the HTML for this LayoutItem.
        """
        if self._html is None:
            self._html = HTML(self.to_json())
        return self._html

    def widget(self) -> Widget:
        """
        Lazily generate & cache the widget for this LayoutItem.
        """
        if self._widget is None:
            self._widget = Widget(self.to_json())
        return self._widget

    def save_html(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(html_standalone(self.to_json()))
        print(f"HTML saved to {path}")

    def save_image(self, path, width=500, height=1000):
        # Save image using headless browser
        hti = Html2Image()
        hti.size = (width, height)
        hti.screenshot(html_str=html_standalone(self.to_json()), save_as=path)
        # Crop transparent regions
        img = Image.open(path)
        content = img.getbbox()
        img = img.crop(content)
        img.save(path)
        print(f"Image saved to {path}")

    def reset(self, other: "LayoutItem") -> None:
        """
        Render a new LayoutItem to this LayoutItem's widget.

        Args:
            new_item: A LayoutItem to reset to.
        """
        display_as = self._display_as or CONFIG["display_as"]
        if display_as != "widget":
            print(
                "Warning: Resetting a non-widget LayoutItem. This will not update the display."
            )
        new_data = other.to_json()
        self.widget().data = new_data
        self.html().data = new_data


class Hiccup(LayoutItem):
    """Wraps a Hiccup-style list to be rendered as an interactive widget in the JavaScript runtime."""

    def __init__(self, *args: Any) -> None:
        LayoutItem.__init__(self)
        if len(args) == 0:
            self.data: list[Any] | tuple[Any, ...] | None = None
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            self.data = args[0]
        else:
            self.data = args

    def to_json(self) -> Any:
        return self.data


def flatten_layout_items(
    items: Sequence[Any], layout_class: type
) -> tuple[list[Any], dict[str, Any]]:
    flattened: list[Any] = []
    options: dict[str, Any] = {}
    for item in items:
        if isinstance(item, layout_class):
            flattened.extend(item.items)
            options.update(item.options)
        elif isinstance(item, dict):
            options.update(item)
        else:
            flattened.append(item)
    return flattened, options


class Row(LayoutItem):
    def __init__(self, *items: Any):
        super().__init__()
        self.items, self.options = flatten_layout_items(items, Row)

    def to_json(self) -> Any:
        return Hiccup(View.Row, self.options, *self.items)


class Column(LayoutItem):
    def __init__(self, *items: Any):
        super().__init__()
        self.items, self.options = flatten_layout_items(items, Column)

    def to_json(self) -> Any:
        return Hiccup(View.Column, self.options, *self.items)


class Slider(LayoutItem):
    def __init__(
        self,
        key: str,
        range: int | Sequence[int],
        label: str | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.config: dict[str, Any] = {
            "state_key": key,
            "range": [0, range] if isinstance(range, int) else range,
            "label": label,
            "kind": "Slider",
            **kwargs,
        }

    def to_json(self) -> Any:
        return View.Reactive(self.config)
