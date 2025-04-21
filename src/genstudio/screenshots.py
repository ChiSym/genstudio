"""
Screenshot utilities for GenStudio plots using a StudioContext which inherits from ChromeContext
"""

import json
import time
import subprocess  # Added import for subprocess
import base64
from pathlib import Path
from typing import Dict, List, Optional, Union

import genstudio.widget as widget
from genstudio.html import encode_buffers
from genstudio.env import WIDGET_URL, CSS_URL
from genstudio.chrome_devtools import ChromeContext, format_bytes
from genstudio.util import read_file


class StudioContext(ChromeContext):
    """
    StudioContext extends ChromeContext with GenStudio-specific methods.
    It encapsulates behavior such as loading the GenStudio environment, rendering plots, and updating state.
    """

    def __init__(self, plot=None, **kwargs):
        """
        Initialize StudioContext with optional plot

        Args:
            plot: Optional plot to load on initialization
            **kwargs: Additional arguments passed to ChromeContext
        """
        self._plot = plot
        super().__init__(**kwargs)

    def __enter__(self):
        context = super().__enter__()
        if self._plot is not None:
            self.load_plot(self._plot)
        return context

    def load_studio_html(self):
        # Check if GenStudio environment is already loaded
        if not self.evaluate("typeof window.genstudio === 'object'"):
            if self.debug:
                print("[screenshots.py] Loading GenStudio HTML")

            files = {}
            # Handle script content based on whether WIDGET_URL is a CDN URL or local file
            if isinstance(WIDGET_URL, str):  # CDN URL
                if self.debug:
                    print(f"[screenshots.py] Using CDN script from: {WIDGET_URL}")
                script_tag = f'<script type="module" src="{WIDGET_URL}"></script>'
            else:  # Local file
                if self.debug:
                    print(f"[screenshots.py] Loading local script from: {WIDGET_URL}")
                script_tag = '<script type="module" src="studio.js"></script>'
                files["studio.js"] = read_file(WIDGET_URL)

            if isinstance(CSS_URL, str):
                style_tag = f'<style>@import "{CSS_URL}";</style>'
            else:
                style_tag = '<style>@import "studio.css";</style>'
                with open(CSS_URL, "r") as file:
                    files["studio.css"] = file.read()

            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset=\"UTF-8\">
                <title>GenStudio</title>
                {style_tag}
                {script_tag}
            </head>
            <body>
                <div id=\"studio\"></div>
            </body>
            </html>
            """
            self.load_html(html, files=files)
        elif self.debug:
            print("GenStudio already loaded, skipping initialization")

    def load_plot(self, plot, measure=True):
        """
        Loads the plot in the GenStudio environment.
        """
        if self.debug:
            print("[StudioContext] Loading plot into GenStudio")

        self.load_studio_html()
        data, buffers = widget.to_json_with_initialState(plot, buffers=[])

        BUFFER_TOTAL_THRESHOLD = 10 * 1024 * 1024  # 10MB total for single URL
        buffers_payload = []

        total_size = sum(len(b) for b in buffers)
        if self.debug:
            print(
                f"[StudioContext] Processing {len(buffers)} buffers for transfer. Total size: {format_bytes(total_size)}"
            )

        if total_size > BUFFER_TOTAL_THRESHOLD:
            # Concatenate all buffers and serve as a single URL
            concat_bytes = b"".join(buffers)
            buffer_filename = f"served_buffer_{self.id}_all.bin"
            self.files[buffer_filename] = concat_bytes
            buffer_url = f"http://localhost:{self.server_port}/{buffer_filename}"
            buffers_payload = {
                "type": "url",
                "url": buffer_url,
                "sizes": [len(b) for b in buffers],
            }
            if self.debug:
                print(
                    f"[StudioContext] Serving all buffers as a single URL: {buffer_url} ({format_bytes(len(concat_bytes))})"
                )
        else:
            for i, buffer_bytes in enumerate(buffers):
                encoded_buffer = base64.b64encode(buffer_bytes).decode("utf-8")
                buffers_payload.append(encoded_buffer)

        if self.debug:
            if isinstance(buffers_payload, dict):
                print("[StudioContext] Transfer summary: all buffers via single URL.")
            else:
                print(f"Total items in buffers payload: {len(buffers_payload)}")

        render_js = f"""
         (async () => {{
           console.log('[StudioContext] Received renderData call for ID: {self.id}');
           const data = {json.dumps(data)};
           // Pass the mixed payload list (base64 strings, or a single URL object)
           const buffers_payload = {json.dumps(buffers_payload)};
           await window.genstudio.renderData('studio', data, buffers_payload, '{self.id}');
           await window.genstudio.whenReady('{self.id}');
         }})()
         """
        self.evaluate(render_js, await_promise=True)

        if measure:
            self.measure_size()

    def measure_size(self):
        """
        Measures container size and adjusts context dimensions accordingly.
        """
        dimensions = self.evaluate("""
            (function() {
                const container = document.querySelector('.genstudio-container');
                if (!container) return null;
                const rect = container.getBoundingClientRect();
                return { width: Math.ceil(rect.width), height: Math.ceil(rect.height) };
            })()
        """)
        if self.debug:
            print(f"[StudioContext] Measured container dimensions: {dimensions}")
        if dimensions is not None:
            self.set_size(dimensions["width"], dimensions["height"])

    def update_state(self, state_updates):
        """
        Sends state updates to GenStudio. Expects state_updates to be a list.
        """
        if self.debug:
            print("[StudioContext] Updating state")
        if not isinstance(state_updates, list):
            raise AssertionError("state_updates must be a list")
        buffers = []
        state_data = widget.to_json(state_updates, buffers=buffers)

        update_js = f"""
        (async function() {{
            try {{
                const updates = {json.dumps(state_data)}
                const buffers = {encode_buffers(buffers)}
                const result = window.genstudio.instances['{self.id}'].updateWithBuffers(updates, buffers);
                await window.genstudio.whenReady('{self.id}');
                return result;
            }} catch (e) {{
                console.error('State update failed:', e);
                return 'error: ' + e.message;
            }}
        }})()
        """
        return self.evaluate(update_js, await_promise=True)

    def capture_image_sequence(self, state_updates: List[Dict]) -> List[bytes]:
        """
        Capture a sequence of images after applying each state update.

        Args:
            state_updates: List of state updates to apply before each capture

        Returns:
            List of PNG image bytes
        """
        bytes_list = []
        for i, state_update in enumerate(state_updates):
            self.update_state([state_update])
            image_bytes = self.capture_image()
            bytes_list.append(image_bytes)
        return bytes_list

    def save_image(
        self,
        output_path: Optional[Union[str, Path]] = None,
        state_update: Optional[Dict] = None,
    ) -> Union[Path, bytes]:
        """
        Save an image of the current plot state.

        Args:
            output_path: Optional path to save the image; if not provided, returns PNG bytes
            state_update: Optional state update to apply before capturing

        Returns:
            Path to saved image if output_path provided, otherwise PNG bytes
        """
        if state_update:
            self.update_state([state_update])
        image_bytes = self.capture_image()

        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(exist_ok=True, parents=True)
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            if self.debug:
                print(f"[StudioContext] Image saved to: {out_path}")
            return out_path
        return image_bytes

    def save_image_sequence(
        self,
        state_updates: List[Dict],
        output_dir: Union[str, Path],
        filenames: Optional[List[str]] = None,
        filename_base: Optional[str] = "screenshot",
    ) -> List[Path]:
        """
        Save a sequence of images after applying each state update.

        Args:
            state_updates: List of state updates to apply before each capture
            output_dir: Directory where images will be saved
            filenames: Optional list of filenames for each image
            filename_base: Base name for generating filenames if not provided

        Returns:
            List of paths to saved images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        if filenames:
            if len(filenames) != len(state_updates):
                raise ValueError(
                    f"Number of filenames ({len(filenames)}) must match number of state updates ({len(state_updates)})"
                )
        else:
            filenames = [f"{filename_base}_{i}.png" for i in range(len(state_updates))]

        output_paths = [output_dir / name for name in filenames]
        saved_paths = []

        image_bytes_list = self.capture_image_sequence(state_updates)
        for i, (image_bytes, out_path) in enumerate(
            zip(image_bytes_list, output_paths)
        ):
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            saved_paths.append(out_path)
            if self.debug:
                print(
                    f"[StudioContext] Saved image {i + 1}/{len(state_updates)} to: {out_path}"
                )

        return saved_paths

    def save_pdf(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> Union[Path, bytes]:
        """
        Save a PDF of the current plot state.

        Args:
            output_path: Optional path to save the PDF; if not provided, returns PDF bytes

        Returns:
            Path to saved PDF if output_path provided, otherwise PDF bytes
        """

        # Trigger WebGPU canvas capture for 3D content before PDF generation
        self.evaluate(f"window.genstudio.beforePDF('{self.id}');", await_promise=True)

        # Capture the PDF content (including static images of 3D canvases)
        pdf_bytes = self.capture_pdf()

        # Cleanup and restore interactive 3D content
        self.evaluate(f"window.genstudio.afterPDF('{self.id}');", await_promise=True)

        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(exist_ok=True, parents=True)
            with open(out_path, "wb") as f:
                f.write(pdf_bytes)
            if self.debug:
                print(f"[StudioContext] PDF saved to: {out_path}")
            return out_path
        return pdf_bytes

    def capture_video(
        self,
        state_updates: List[Dict],
        filename: Union[str, Path],
        fps: int = 24,
    ) -> Path:
        """
        Capture a series of states from a plot as a video.
        The video is generated without saving intermediate images to disk by piping PNG frames
        directly to ffmpeg.

        Args:
            state_updates: List of state update dictionaries to apply sequentially
            filename: Path where the resulting video will be saved
            fps: Frame rate (frames per second) for the video

        Returns:
            Path to the saved video file
        """
        filename = Path(filename)
        if self.debug:
            print(f"[StudioContext] Recording video with {len(state_updates)} frames")

        start_time = time.time()
        filename.parent.mkdir(exist_ok=True, parents=True)

        # Detect file extension
        ext = filename.suffix.lower()

        if ext == ".gif":
            ffmpeg_cmd = (
                f"ffmpeg {'-v error' if not self.debug else ''} -y "
                f"-f image2pipe -vcodec png -framerate {fps} -i - "
                # The filter below: (1) splits the pipeline into two streams;
                #                   (2) generates a palette from one stream;
                #                   (3) applies that palette to the other stream;
                #                   (4) loops infinitely (0) in the final GIF.
                f'-vf "split [a][b];[b]palettegen=stats_mode=diff[p];[a][p]paletteuse=new=1" '
                f'-c:v gif -loop 0 "{filename}"'
            )
        else:
            # Fallback: generate MP4 video with libx264
            ffmpeg_cmd = (
                f"ffmpeg {'-v error' if not self.debug else ''} -y "
                f"-f image2pipe -vcodec png -r {fps} -i - "
                f'-an -c:v libx264 -pix_fmt yuv420p "{filename}"'
            )

        if self.debug:
            print(f"[StudioContext] Running ffmpeg command: {ffmpeg_cmd}")

        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, shell=True)

        try:
            for i, state_update in enumerate(state_updates):
                result = self.update_state([state_update])
                if self.debug:
                    print(f"[StudioContext] State update {i} result: {result}")
                frame_bytes = self.capture_image()
                if proc.stdin:
                    proc.stdin.write(frame_bytes)
                    if self.debug:
                        print(f"[StudioContext] Captured frame {i}")

            if proc.stdin:
                proc.stdin.close()
            proc.wait()

            elapsed_time = time.time() - start_time
            actual_fps = len(state_updates) / elapsed_time
            if self.debug:
                print(
                    f"[StudioContext] Video generation took {elapsed_time:.2f} seconds (~{actual_fps:.1f} fps)"
                )

            return filename
        except Exception as e:
            # Clean up process on error
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            raise e


def save_image(
    plot,
    output_path: Optional[Union[str, Path]] = None,
    state_update: Optional[Dict] = None,
    width: int = 400,
    height: Optional[int] = None,
    scale: float = 1.0,
    debug: bool = False,
) -> Union[Path, bytes]:
    """
    Render the plot and capture an image.

    Args:
        plot: The GenStudio plot widget
        output_path: Optional path to save the image; if not provided, returns PNG bytes
        state_update: Optional state update to apply before capture
        width: Width of the browser window
        height: Optional height of the browser window
        scale: Device scale factor
        debug: Whether to print debug information

    Returns:
        Path to saved image if output_path is provided, otherwise PNG bytes
    """
    with StudioContext(
        plot=plot, width=width, height=height, scale=scale, debug=debug
    ) as studio:
        return studio.save_image(output_path, state_update)


def save_images(
    plot,
    state_updates: List[Dict],
    output_dir: Union[str, Path] = "./scratch/screenshots",
    filenames: Optional[List[str]] = None,
    filename_base: Optional[str] = "screenshot",
    width: int = 800,
    height: Optional[int] = None,
    scale: float = 1.0,
    debug: bool = False,
) -> List[Path]:
    """
    Capture a sequence of images with state updates.

    Args:
        plot: The GenStudio plot widget
        state_updates: List of state update dictionaries to apply sequentially
        output_dir: Directory where images will be saved
        filenames: Optional list of filenames for each image; if not provided, filenames will be auto-generated
        filename_base: Base name for generating filenames
        width: Width of the browser window
        height: Optional height of the browser window
        scale: Device scale factor
        debug: Whether to print debug information

    Returns:
        List of paths to the saved images
    """
    with StudioContext(
        plot=plot, width=width, height=height, scale=scale, debug=debug
    ) as studio:
        return studio.save_image_sequence(
            state_updates, output_dir, filenames, filename_base
        )


def save_pdf(
    plot,
    output_path: Optional[Union[str, Path]] = None,
    width: int = 400,
    height: Optional[int] = None,
    scale: float = 1.0,
    debug: bool = False,
) -> Union[Path, bytes]:
    """
    Render the plot and capture a PDF of the page.

    Args:
        plot: The GenStudio plot widget
        output_path: Optional path to save the PDF; if not provided, returns PDF bytes
        width: Width of the browser window
        height: Optional height of the browser window
        scale: Device scale factor
        debug: Whether to print debug information

    Returns:
        Path to saved PDF if output_path is provided, otherwise PDF bytes
    """
    with StudioContext(
        plot=plot, width=width, height=height, scale=scale, debug=debug
    ) as studio:
        return studio.save_pdf(output_path)


def save_video(
    plot,
    state_updates: List[Dict],
    filename: Union[str, Path],
    fps: int = 24,
    width: int = 400,
    height: Optional[int] = None,
    scale: float = 1.0,
    debug: bool = False,
) -> Path:
    """
    Capture a series of states from a plot as a video.

    Args:
        plot: The GenStudio plot widget
        state_updates: List of state update dictionaries to apply sequentially
        filename: Path where the resulting video will be saved
        fps: Frame rate (frames per second) for the video
        width: Width of the browser window
        height: Optional height of the browser window
        scale: Device scale factor
        debug: Whether to print debug information

    Returns:
        Path to the saved video file
    """
    with StudioContext(
        plot=plot, width=width, height=height, scale=scale, debug=debug
    ) as studio:
        return studio.capture_video(state_updates, filename, fps)
