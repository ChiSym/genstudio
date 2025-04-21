# chrome_devtools.py
"""
Simple Chrome DevTools Protocol client for HTML content manipulation and screenshots
"""

import os
import time
import json
import base64
import shutil
import subprocess
import urllib.request
from websockets.sync.client import connect
import http.server
import socketserver
import sys
import threading
from pathlib import Path
from typing import Union

DEBUG_WINDOW = False


def format_bytes(bytes):
    if bytes >= 1024 * 1024:
        return f"{bytes / (1024 * 1024):.2f}MB"
    return f"{bytes / 1024:.2f}KB"


def find_chrome():
    """Find Chrome executable on the system"""
    possible_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # macOS
        "/usr/bin/google-chrome",  # Linux
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",  # Windows
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]

    # Check PATH first
    for cmd in ["google-chrome", "chromium", "chromium-browser", "chrome"]:
        chrome_path = shutil.which(cmd)
        if chrome_path:
            return chrome_path

    # Check common installation paths
    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("Could not find Chrome. Please install Chrome.")


def check_chrome_version(chrome_path):
    """Check if Chrome version supports the new headless mode

    Args:
        chrome_path: Path to Chrome executable

    Returns:
        tuple: (version_number, is_supported)

    Raises:
        RuntimeError: If Chrome version cannot be determined
    """
    try:
        # Run Chrome with --version flag
        output = subprocess.check_output(
            [chrome_path, "--version"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        # Parse version string (format like "Google Chrome 112.0.5615.49")
        version_str = output.strip()
        # Extract version number
        import re

        match = re.search(r"(\d+)\.", version_str)
        if not match:
            raise RuntimeError(f"Could not parse Chrome version from: {version_str}")

        major_version = int(match.group(1))
        # New headless mode (--headless=new) was introduced in Chrome 109
        return major_version, major_version >= 109
    except subprocess.SubprocessError as e:
        raise RuntimeError(f"Failed to determine Chrome version: {e}")


# Start HTTP server first
class DualDirectoryHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, chrome_context, **kwargs):
        self._logged_paths = set()
        self.chrome_context = chrome_context
        super().__init__(*args, **kwargs)

    def guess_type(self, path):
        """Guess the type of a file based on its extension"""
        ext = str(path).split(".")[-1].lower()
        if ext == "html":
            return "text/html"
        elif ext == "js":
            return "application/javascript"
        elif ext == "css":
            return "text/css"
        return "text/plain"

    def translate_path(self, path):
        # Special handling for buffers served from memory
        if path.lstrip("/").startswith("served_buffer_"):
            # Keep the path as is, do_GET will handle it
            return path.lstrip("/")

        # First try other served files
        if path.lstrip("/") in self.chrome_context.files:
            if self.chrome_context.debug and path not in self._logged_paths:
                print(f"[chrome_devtools.py] Serving {path} from memory")
                self._logged_paths.add(path)
            return path
        # Fall back to cwd
        cwd_path = os.path.join(os.getcwd(), path.lstrip("/"))
        if os.path.exists(cwd_path) and path not in self._logged_paths:
            if self.chrome_context.debug:
                print(f"[chrome_devtools.py] Serving {path} from cwd: {cwd_path}")
                self._logged_paths.add(path)
        return cwd_path

    def do_GET(self):
        path = self.path.lstrip("/")
        # Check if it's a request for a served buffer
        if path.startswith("served_buffer_") and path in self.chrome_context.files:
            # Serve buffer as raw bytes and clean up
            content_bytes = self.chrome_context.files.pop(
                path
            )  # Use pop to get and remove
            if self.chrome_context.debug:
                print(
                    f"[chrome_devtools.py] Serving & Deleting buffer {path} ({len(content_bytes)} bytes) from memory"
                )
            self.send_response(200)
            self.send_header("Content-type", "application/octet-stream")
            self.send_header("Content-Length", str(len(content_bytes)))
            # Add CORS header to allow fetching from JS
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content_bytes)
            return

        # Handle index.html and other explicitly served files
        if path in self.chrome_context.files:
            content = self.chrome_context.files[path]
            self.send_response(200)
            self.send_header("Content-type", self.guess_type(path))
            self.send_header("Content-Length", str(len(content.encode())))
            self.end_headers()
            self.wfile.write(content.encode())
            return
        return super().do_GET()

    def log_message(self, format, *args):
        # Suppress default logging
        pass


class ChromeContext:
    """Manages a Chrome instance and provides methods for content manipulation and screenshots"""

    def __init__(self, port=9222, width=400, height=None, scale=1.0, debug=False):
        self.id = f"chrome_{int(time.time() * 1000)}_{hash(str(port))}"  # Unique ID for this context
        self.port = port
        self.width = width
        self.height = height
        self.scale = scale
        self.debug = debug
        self.chrome_process = None
        self.ws = None
        self.cmd_id = 0
        self.files = {}
        self.httpd = None
        self.server_thread = None
        self.server_port = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def set_size(self, width=None, height=None, scale=None):
        self.width = width or self.width
        self.height = height or self.height or self.width
        if scale:
            self.scale = scale
        if DEBUG_WINDOW:
            self._send_command(
                "Browser.setWindowBounds",
                {
                    "windowId": self._send_command("Browser.getWindowForTarget")[
                        "windowId"
                    ],
                    "bounds": {"width": self.width, "height": self.height},
                },
            )
        self._send_command(
            "Page.setDeviceMetricsOverride",
            {
                "width": self.width,
                "height": self.height,
                "deviceScaleFactor": self.scale,
                "mobile": False,
            },
        )

    def start(self):
        """Start Chrome and connect to DevTools Protocol"""
        if self.chrome_process:
            if self.debug:
                print(
                    "[chrome_devtools.py] Chrome already started, adjusting size only"
                )
            self.set_size()
            return  # Already started

        self.httpd = socketserver.TCPServer(
            ("localhost", 0),
            lambda *args: DualDirectoryHandler(*args, chrome_context=self),
        )
        self.server_port = self.httpd.server_address[1]

        if self.debug:
            print(
                f"[chrome_devtools.py] Starting HTTP server on port {self.server_port}"
            )

        self.server_thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.1}
        )
        self.server_thread.daemon = True
        self.server_thread.start()

        chrome_path = find_chrome()
        if self.debug:
            print(f"[chrome_devtools.py] Starting Chrome from: {chrome_path}")

        # Check Chrome version for headless mode compatibility
        version, supports_new_headless = check_chrome_version(chrome_path)
        if self.debug:
            print(f"[chrome_devtools.py] Chrome version: {version}")
            if not supports_new_headless:
                print(
                    f"[chrome_devtools.py] Warning: Chrome version {version} does not support the new headless mode (--headless=new). Using legacy headless mode instead"
                )

        # Determine appropriate headless flag
        headless_flag = ""
        if not DEBUG_WINDOW:
            headless_flag = "--headless=new" if supports_new_headless else "--headless"

        # Base Chrome flags
        chrome_cmd = [
            chrome_path,
            headless_flag,
            f"--remote-debugging-port={self.port}",
            "--remote-allow-origins=*",
            "--disable-search-engine-choice-screen",
            "--ash-no-nudges",
            "--no-first-run",
            "--disable-features=Translate",
            "--no-default-browser-check",
            "--hide-scrollbars",
            f"--window-size={self.width},{self.height or self.width}",
            "--app=data:,",
        ]

        # Add Linux-specific WebGPU flags
        if sys.platform.startswith("linux"):
            chrome_cmd.extend(
                [
                    "--no-sandbox",
                    "--use-angle=vulkan",
                    "--enable-features=Vulkan",
                    "--enable-unsafe-webgpu",
                ]
            )

        self.chrome_process = subprocess.Popen(
            chrome_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Wait for Chrome to start by polling
        start_time = time.time()
        if self.debug:
            print(
                f"[chrome_devtools.py] Attempting to connect to Chrome on port {self.port}"
            )

        while True:
            try:
                response = urllib.request.urlopen(f"http://localhost:{self.port}/json")
                targets = json.loads(response.read())
                page_target = next(
                    (
                        t
                        for t in targets
                        if t["type"] == "page" and t["url"] == "data:,"
                    ),
                    None,
                )
                if page_target:
                    if self.debug:
                        print("[chrome_devtools.py] Successfully found Chrome target")
                    break
            except Exception:
                pass

            if time.time() - start_time > 10:
                raise RuntimeError("Chrome did not start in time")

        # Connect to the page target
        self.ws = connect(page_target["webSocketDebuggerUrl"])
        # Enable required domains
        self._send_command("Page.enable")
        self._send_command("Runtime.enable")
        self._send_command("Console.enable")  # Enable console events

    def stop(self):
        """Stop Chrome and clean up"""
        if self.debug:
            print("[chrome_devtools.py] Stopping Chrome process")

        if self.ws:
            self.ws.close()
            self.ws = None

        if self.chrome_process and not DEBUG_WINDOW:
            self.chrome_process.terminate()
            try:
                self.chrome_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if self.debug:
                    print(
                        "[chrome_devtools.py] Chrome process did not terminate, forcing kill"
                    )
                self.chrome_process.kill()
            self.chrome_process = None

        if self.httpd:
            if self.debug:
                print("[chrome_devtools.py] Shutting down HTTP server")
            self.httpd.shutdown()
            if self.server_thread:
                self.server_thread.join()
            self.httpd.server_close()
            self.httpd = None
            self.server_thread = None
            self.server_port = None

    def _send_command(self, method, params=None):
        """Send a command to Chrome and wait for the response"""
        if not self.ws:
            raise RuntimeError("Not connected to Chrome")

        self.cmd_id += 1
        message = {"id": self.cmd_id, "method": method, "params": params or {}}

        message_str = json.dumps(message)
        # if self.debug:
        #     size_bytes = len(message_str.encode("utf-8"))
        #     print(
        #         f"[chrome_devtools.py] Sending message of size: {format_bytes(size_bytes)} via WebSocket"
        #     )

        self.ws.send(message_str)

        # Wait for response with matching id
        while True:
            response = json.loads(self.ws.recv())

            # Print console messages if debug is enabled
            if self.debug and response.get("method") == "Console.messageAdded":
                message = response["params"]["message"]
                level = message.get("level", "log")
                text = message.get("text", "")
                print(f"[chrome.{level}]: {text}")

            # Handle command response
            if "id" in response and response["id"] == self.cmd_id:
                if "error" in response:
                    raise RuntimeError(
                        f"Chrome DevTools command failed: {response['error']}"
                    )
                return response.get("result", {})

    def load_html(self, html, files=None):
        """Serve HTML content and optional files over localhost and load it in the page"""
        self.set_size()

        # Update files dictionary
        if files:
            self.files.update(files)
        self.files["index.html"] = html

        # Navigate to page
        url = f"http://localhost:{self.server_port}/index.html"
        self._send_command("Page.navigate", {"url": url})

        while True:
            if not self.ws:
                raise RuntimeError("[chrome_devtools.py] WebSocket connection lost")
            response = json.loads(self.ws.recv())
            if response.get("method") == "Page.loadEventFired":
                if self.debug:
                    print("[chrome_devtools.py] Page load complete")
                break

    def evaluate(self, expression, return_by_value=True, await_promise=False):
        """Evaluate JavaScript code in the page context

        Args:
            expression: JavaScript expression to evaluate
            return_by_value: Whether to return the result by value
            await_promise: Whether to wait for promise resolution
        """
        result = self._send_command(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": return_by_value,
                "awaitPromise": await_promise,
            },
        )

        return result.get("result", {}).get("value")

    def capture_image(self) -> bytes:
        """Capture a screenshot of the page as PNG bytes."""
        if self.debug:
            print("[chrome_devtools.py] Capturing image")

        result = self._send_command(
            "Page.captureScreenshot",
            {
                "format": "png",
                "captureBeyondViewport": True,
                "clip": {
                    "x": 0,
                    "y": 0,
                    "width": self.width,
                    "height": self.height,
                    "scale": self.scale,
                },
            },
        )

        if not result or "data" not in result:
            raise RuntimeError("Failed to capture image")

        return base64.b64decode(result["data"])

    def capture_pdf(self) -> bytes:
        """Capture the current page as a PDF and return PDF bytes."""
        if self.debug:
            print("[chrome_devtools.py] Capturing PDF")

        # Convert pixel width to inches at 96 DPI
        paper_width = self.width / 96
        paper_height = paper_width * ((self.height or self.width) / self.width)

        # Request PDF with stream transfer mode
        result = self._send_command(
            "Page.printToPDF",
            {
                "landscape": False,
                "printBackground": True,
                "preferCSSPageSize": True,
                "paperWidth": paper_width,
                "paperHeight": paper_height,
                "marginTop": 0,
                "marginBottom": 0,
                "marginLeft": 0,
                "marginRight": 0,
                "transferMode": "ReturnAsStream",
            },
        )
        if not result or "stream" not in result:
            raise RuntimeError("Failed to capture PDF - no stream handle returned")

        # Read the PDF data in chunks
        stream_handle = result["stream"]
        pdf_chunks = []

        while True:
            chunk_result = self._send_command(
                "IO.read", {"handle": stream_handle, "size": 500000}
            )

            if not chunk_result:
                raise RuntimeError("Failed to read PDF stream")

            if "data" in chunk_result:
                pdf_chunks.append(base64.b64decode(chunk_result["data"]))

            if chunk_result.get("eof", False):
                break

        # Close the stream
        self._send_command("IO.close", {"handle": stream_handle})

        # Combine all chunks
        return b"".join(pdf_chunks)

    def check_webgpu_support(self):
        """Check if WebGPU is available and functional in the browser

        Returns:
            dict: Detailed WebGPU support information including:
                - supported: bool indicating if WebGPU is available
                - adapter: information about the GPU adapter if available
                - reason: explanation if WebGPU is not supported
                - features: list of supported features if available
        """
        # First load a blank page to ensure we have a proper context
        self.load_html("<html><body></body></html>")

        result = self.evaluate(
            """
            (async function() {
                if (!navigator.gpu) {
                    return {
                        supported: false,
                        reason: 'navigator.gpu is not available'
                    };
                }

                try {
                    // Request adapter with power preference to ensure we get a GPU
                    const adapter = await navigator.gpu.requestAdapter({
                        powerPreference: 'high-performance'
                    });

                    if (!adapter) {
                        return {
                            supported: false,
                            reason: 'No WebGPU adapter found'
                        };
                    }
                    // note that adapter.requestAdapterInfo doesn't always exist so we don't use it

                    // Request device with basic features
                    const device = await adapter.requestDevice({
                        requiredFeatures: []
                    });

                    if (!device) {
                        return {
                            supported: false,
                            reason: 'Failed to create WebGPU device'
                        };
                    }

                    // Try to create a simple buffer to verify device works
                    try {
                        const testBuffer = device.createBuffer({
                            size: 4,
                            usage: GPUBufferUsage.COPY_DST
                        });
                        testBuffer.destroy();
                    } catch (e) {
                        return {
                            supported: false,
                            reason: 'Device creation succeeded but buffer operations failed'
                        };
                    }

                    return {
                        supported: true,
                        adapter: {
                            name: 'WebGPU Device'
                        },
                        features: Array.from(adapter.features).map(f => f.toString())
                    };
                } catch (e) {
                    return {
                        supported: false,
                        reason: e.toString()
                    };
                }
            })()
        """,
            await_promise=True,
        )

        if self.debug:
            if result.get("supported"):
                print(
                    f"[chrome_devtools.py] WebGPU Adapter: '{result.get('adapter', {}).get('name')}'"
                )
                print(
                    f"[chrome_devtools.py]   Features: {', '.join(result.get('features', []))}"
                )
            else:
                print(
                    f"[chrome_devtools.py] WebGPU not supported: {result.get('reason')}"
                )

        return result

    def save_gpu_info(self, output_path: Union[str, Path]):
        """Save Chrome's GPU diagnostics page (chrome://gpu) to a PDF file

        Args:
            output_path: Path where to save the PDF file

        Returns:
            Path to the saved PDF file
        """
        output_path = Path(output_path)
        if self.debug:
            print(f"[chrome_devtools.py] Capturing GPU diagnostics to: {output_path}")

        # Navigate to GPU info page
        self._send_command("Page.navigate", {"url": "chrome://gpu"})

        # Wait for page load
        while True and self.ws:
            response = json.loads(self.ws.recv())
            if response.get("method") == "Page.loadEventFired":
                break

        # Print to PDF
        result = self._send_command(
            "Page.printToPDF",
            {
                "landscape": False,
                "printBackground": True,
                "preferCSSPageSize": True,
            },
        )

        if not result or "data" not in result:
            raise RuntimeError("Failed to generate PDF")

        # Save PDF
        pdf_data = base64.b64decode(result["data"])
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, "wb") as f:
            f.write(pdf_data)

        if self.debug:
            print(f"[chrome_devtools.py] GPU diagnostics saved to: {output_path}")

        return output_path


def main():
    """Example usage"""
    html = """
    <html>
    <head></head>
    <body style="background:red; width:100vw; height:100vh;"><div></div></body>
    </html>
    """

    with ChromeContext(width=400, height=600, debug=True) as chrome:
        # Check WebGPU support
        chrome.check_webgpu_support()

        # Load content served via localhost
        chrome.load_html(html)

        # Capture and save red background image
        image_data = chrome.capture_image()
        Path("./scratch/screenshots").mkdir(exist_ok=True, parents=True)
        with open("./scratch/screenshots/webgpu_test_red.png", "wb") as f:
            f.write(image_data)

        # Change to green and capture again
        chrome.evaluate('document.body.style.background = "green"; "changed!"')
        image_data = chrome.capture_image()
        with open("./scratch/screenshots/webgpu_test_green.png", "wb") as f:
            f.write(image_data)


if __name__ == "__main__":
    main()
