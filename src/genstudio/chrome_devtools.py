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
import websocket

DEBUG = False


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


class ChromeContext:
    """Manages a Chrome instance and provides methods for content manipulation and screenshots"""

    def __init__(self, port=9222, window_size=(400, 400), window_scale=2):
        self.port = port
        self.window_size = window_size
        self.window_scale = window_scale
        self.chrome_process = None
        self.ws = None
        self.cmd_id = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        """Start Chrome and connect to DevTools Protocol"""
        if self.chrome_process:
            return  # Already started

        chrome_path = find_chrome()
        chrome_cmd = [
            chrome_path,
            f"--remote-debugging-port={self.port}",
            "--remote-allow-origins=*",
            "--headless=new" if not DEBUG else "",
            "--no-sandbox",
            "--hide-scrollbars",
            f"--window-size={self.window_size[0]},{self.window_size[1]}",
            "about:blank",
        ]

        self.chrome_process = subprocess.Popen(
            chrome_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Wait for Chrome to start by polling every 100ms
        start_time = time.time()
        while True:
            try:
                response = urllib.request.urlopen(f"http://localhost:{self.port}/json")
                targets = json.loads(response.read())
                page_target = next((t for t in targets if t["type"] == "page"), None)
                if page_target:
                    break
            except Exception:
                pass

            if time.time() - start_time > 10:  # Timeout after 10 seconds
                raise RuntimeError("Chrome did not start in time")

        # Connect to the page target
        self.ws = websocket.create_connection(page_target["webSocketDebuggerUrl"])

        # Enable required domains
        self._send_command("Page.enable")
        self._send_command("Runtime.enable")

    def stop(self):
        """Stop Chrome and clean up"""
        if self.ws:
            self.ws.close()
            self.ws = None

        if self.chrome_process and not DEBUG:
            self.chrome_process.terminate()
            try:
                self.chrome_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.chrome_process.kill()
            self.chrome_process = None

    def _send_command(self, method, params=None):
        """Send a command to Chrome and wait for the response"""
        if not self.ws:
            raise RuntimeError("Not connected to Chrome")

        self.cmd_id += 1
        message = {"id": self.cmd_id, "method": method, "params": params or {}}

        self.ws.send(json.dumps(message))

        # Wait for response with matching id
        while True:
            response = json.loads(self.ws.recv())
            if "id" in response and response["id"] == self.cmd_id:
                if "error" in response:
                    raise RuntimeError(
                        f"Chrome DevTools command failed: {response['error']}"
                    )

                return response.get("result", {})

    def set_content(self, html):
        """Load HTML content into the page"""
        # Navigate to blank page first
        self._send_command("Page.navigate", {"url": "about:blank"})

        # Wait for page load
        while True:
            if not self.ws:
                raise RuntimeError("WebSocket connection lost")
            response = json.loads(self.ws.recv())
            if response.get("method") == "Page.loadEventFired":
                break

        # Set the content
        self._send_command(
            "Runtime.evaluate",
            {
                "expression": f"""
                document.open();
                document.write({repr(html)});
                document.close();
            """,
                "awaitPromise": True,
            },
        )

        # Wait for the page to signal readiness via window.fireWhenReady promise
        self._send_command(
            "Runtime.evaluate",
            {
                "expression": """
                (async () => {
                    while (!window.fireWhenReady) {
                        await new Promise(resolve => setTimeout(resolve, 10));
                    }
                    return window.fireWhenReady;
                })()
                """,
                "awaitPromise": True,
            },
        )

    def evaluate(self, expression, return_by_value=True):
        """Evaluate JavaScript code in the page context"""
        result = self._send_command(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": return_by_value},
        )
        return result.get("result", {}).get("value")

    def screenshot(self, path=None):
        """Take a screenshot of the page"""
        result = self._send_command(
            "Page.captureScreenshot",
            {
                "format": "png",
                "clip": {
                    "x": 0,
                    "y": 0,
                    "width": self.window_size[0],
                    "height": self.window_size[1],
                    "scale": self.window_scale,
                },
            },
        )

        if not result or "data" not in result:
            raise RuntimeError("Failed to capture screenshot")

        image_data = base64.b64decode(result["data"])
        if path:
            with open(path, "wb") as f:
                f.write(image_data)
            return path
        return image_data


def main():
    """Example usage"""
    html = """
    <html>
    <head>
        <style>
            body { margin: 0; }
            div { width: 100vw; height: 100vh; background: red; }
        </style>
    </head>
    <body><div></div></body>
    </html>
    """

    with ChromeContext() as chrome:
        # Load content
        chrome.set_content(html)

        # Evaluate some JavaScript
        result = chrome.evaluate('document.body.style.background = "green"; "changed!"')
        print("Eval result:", result)

        # Take a screenshot
        chrome.screenshot("test.png")


if __name__ == "__main__":
    main()
