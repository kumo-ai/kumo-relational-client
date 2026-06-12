import logging
import os
import platform
from datetime import datetime

from kumoai import in_notebook


def authenticate(api_url: str | None = None) -> None:
    """Authenticates the user and sets the Kumo API key for the SDK.

    This function detects the current environment and launches the appropriate
    authentication flow:
    - In Google Colab: displays an interactive widget to generate and set the
    API key.
    - In all other environments: opens a browser for OAuth2 login, or allows
    manual API key entry if browser login fails.

    After successful authentication, the API key is set in the "KUMO_API_KEY"
    environment variable for use by the SDK.

    Args:
        api_url (str, optional): The base URL for the Kumo API
        (e.g., 'https://kumorfm.ai'). If not provided, uses the
        'KUMO_API_URL' environment variable.
    """
    import re

    if api_url is None:
        api_url = os.getenv("KUMO_API_URL", "https://kumorfm.ai")

    # Remove everything after the domain (keep protocol and domain)
    # e.g. https://kumorfm.ai/api/xyz -> https://kumorfm.ai
    # This is needed to create API keys using the UI popup flow
    api_url = re.sub(
        r"(https?://[^/]+).*", r"\1",
        api_url.rstrip('/')) if '://' in api_url else api_url.split('/')[0]

    from importlib.util import find_spec
    if find_spec("google.colab") is not None:
        _authenticate_colab(api_url)
    else:
        _authenticate_local(api_url)


def _authenticate_local(api_url: str, redirect_port: int = 8765) -> None:
    """Starts an HTTP server on the user's local machine to handle OAuth2
    or similar login flow, opens the browser for user login, and sets the
    API key via the "KUMO_API_KEY" environment variable.

    If browser-based authentication fails or is not possible, allows the
    user to manually paste an API key.

    Args:
        api_url (str): The base URL for authentication (login page).
        redirect_port (int, optional): The port for the local callback
        server (default: 8765).
    """
    import http.server
    import threading
    import time
    import urllib.parse
    import webbrowser
    from getpass import getpass
    from socketserver import TCPServer
    from typing import Any

    logger = logging.getLogger('kumoai')

    token_status: dict[str, Any] = {
        'token': None,
        'token_name': None,
        'failed': False
    }

    token_name = (f"sdk-{platform.node().lower()}-" +
                  datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-Z')

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed_path = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed_path.query)
            token = params.get('token', [None])[0]
            received_token_name = params.get('token_name', [None])[0]

            if token:
                token_status['token'] = token
                token_status['token_name'] = received_token_name
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
            else:
                token_status['failed'] = True
                self.send_response(400)
                self.end_headers()

            html = f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authenticate SDK</title>
                <style>
                    body {{
                        margin: 0;
                        padding: 0;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        font-family:
                            -apple-system,
                            BlinkMacSystemFont,
                            'Segoe UI', Roboto, sans-serif;
                    }}
                    .container {{
                        text-align: center;
                        padding: 40px;
                    }}
                    svg {{
                        margin-bottom: 20px;
                    }}
                    p {{
                        font-size: 18px;
                        color: #333;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <?xml version="1.0" encoding="UTF-8"?>
                    <svg xmlns="http://www.w3.org/2000/svg"
                        id="kumo-logo" width="183.908" height="91.586"
                        viewBox="0 0 183.908 91.586">
                        <g id="c">
                            <g id="Group_9893" data-name="Group 9893">
                                <path id="Path_4831" data-name="Path 4831"
                                    d="M67.159,67.919V46.238L53.494,59.491,
                                    68.862,82.3H61.567L49.1,63.74l-7.011,6.8V82.3h-6.02V29.605h6.02V62.182l16.642-16.36H73.109v22.1c0,5.453,3.611,9.419,9.277,9.419,5.547,0,9.14-3.9,9.2-9.282V0H0V91.586H91.586V80.317a15.7,15.7,0,0,1-9.2,2.828c-8.569,0-15.226-6.02-15.226-15.226Z"
                                    fill="#d40e8c">
                                </path>
                                <path id="Path_4832" data-name="Path 4832"
                                    d="M233.452,121.881h-6.019V98.3c0-4.745-3.117-8.286-7.932-8.286s-7.932,3.541-7.932,8.286v23.583h-6.02V98.3c0-4.745-3.116-8.286-7.932-8.286s-7.932,3.541-7.932,8.286v23.583h-6.02V98.51c0-7.932,5.736-14.023,13.952-14.023a12.106,12.106,0,0,1,10.906,6.02,12.3,12.3,0,0,1,10.978-6.02c8.285,0,13.951,6.091,13.951,14.023v23.37Z"
                                    transform="translate(-86.054 -39.585)"
                                    fill="#d40e8c">
                                </path>
                                <path id="Path_4833" data-name="Path 4833"
                                    d="M313.7,103.751c0,10.481-7.932,
                                    19.051-18.342,19.051-10.341,
                                    0-18.343-8.569-18.343-19.051,0-10.623,
                                    8-19.263,18.343-19.263C305.767,84.488,
                                    313.7,93.128,313.7,103.751Zm-6.02,
                                    0c0-7.436-5.523-13.527-12.322-13.527-6.728
                                    ,0-12.252,6.091-12.252,13.527,0,7.295,
                                    5.524,13.244,12.252,13.244,6.8,0,
                                    12.322-5.949,12.322-13.244Z"
                                    transform="translate(-129.791 -39.585)"
                                    fill="#d40e8c">
                                </path>
                            </g>
                        </g>
                    </svg>

                    <div id="success-div"
                        style="background: #f2f8f0;
                            border: 1px solid #1d8102;
                            border-radius: 1px;
                            padding: 24px 32px;
                            margin: 24px auto 0 auto;
                            max-width: 400px;
                            text-align: left;
                            display: none;"
                    >
                        <div style="font-size: 1.1em;
                            font-weight: bold;
                            margin-bottom: 10px;
                            text-align: left;"
                        >
                            Request successful
                        </div>
                        <div style="font-size: 1.1em;">
                            Kumo SDK has been granted a token.
                            You may now close this window.
                        </div>
                    </div>

                    <div id="failure-div"
                        style="background: #ffebeb;
                            border: 1px solid #ff837a;
                            border-radius: 1px;
                            padding: 24px 32px;
                            margin: 24px auto 0 auto;
                            max-width: 400px;
                            text-align: left;
                            display: none;"
                    >
                        <div style="font-size: 1.1em;
                            font-weight: bold;
                            margin-bottom: 10px;
                            text-align: left;"
                        >
                            Request failed
                        </div>
                        <div style="font-size: 1.1em;">
                            Failed to generate a token.
                            Please try manually creating a token at
                                <a href="{api_url}/api-keys" target="_blank">
                                    {api_url}/api-keys
                                </a>
                            or contact Kumo for further assistance.
                        </div>
                    </div>

                    <script>
                        // Show only the appropriate div based on the result
                        const search = window.location.search;
                        const urlParams = new URLSearchParams(search);
                        const hasToken = urlParams.has('token');
                        if (hasToken) {{
                            document
                                .getElementById('success-div')
                                .style.display = 'block';
                        }} else {{
                            document
                                .getElementById('failure-div')
                                .style.display = 'block';
                        }}
                    </script>
                </div>
            </body>
            </html>
            '''
            self.wfile.write(html.encode('utf-8'))

        def log_message(self, format: str, *args: object) -> None:
            return  # Suppress logging

    # Find a free port if needed
    port = redirect_port
    for _ in range(10):
        try:
            with TCPServer(("", port), CallbackHandler) as _:
                break
        except OSError:
            port += 1
    else:
        raise RuntimeError(
            "Could not find a free port for the callback server.")

    # Start the server in a thread
    def serve() -> None:
        with TCPServer(("", port), CallbackHandler) as httpd:
            httpd.timeout = 60
            while token_status['token'] is None:
                httpd.handle_request()

    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()

    # Construct the login URL with callback_url and token_name
    callback_url = f"http://127.0.0.1:{port}/"
    login_url = (f"{api_url}/authenticate-sdk/" +
                 f"?callback_url={urllib.parse.quote(callback_url)}" +
                 f"&token_name={urllib.parse.quote(token_name)}")

    print(
        "Opening browser page to automatically generate an API key...\n" +
        "If the page does not open, manually create a new API key at " +
        f"{api_url}/api-keys and set it using os.environ[\"KUMO_API_KEY\"] " +
        "= \"YOUR_API_KEY\"")

    webbrowser.open(login_url)

    def get_user_input() -> None:
        token_entered = getpass(
            "or paste the API key here and press enter: ").strip()

        while (len(token_entered) == 0):
            token_entered = getpass(
                "API Key (type then press enter): ").strip()

        token_status['token'] = token_entered

    if not in_notebook():
        user_input_thread = threading.Thread(target=get_user_input,
                                             daemon=True)
        user_input_thread.start()

    # Wait for the token (timeout after 120s)
    start = time.time()
    while token_status['token'] is None and time.time() - start < 120:
        time.sleep(1)

    if not isinstance(token_status['token'], str) or not token_status['token']:
        raise TimeoutError(
            "Timed out waiting for authentication or API key input.")

    os.environ['KUMO_API_KEY'] = token_status['token']

    logger.info(
        f"Generated token \"{token_status['token_name'] or token_name}\" " +
        "and saved to KUMO_API_KEY env variable")


def _authenticate_colab(api_url: str) -> None:
    """Displays an interactive widget in Google Colab to authenticate the user
    and generate a Kumo API key.

    This method is intended to be used within a Google Colab notebook. It
    presents a button that, when clicked, opens a popup for the user to
    authenticate with KumoRFM and generate an API key. Upon successful
    authentication, the API key is set in the notebook's environment using the
    "KUMO_API_KEY" variable. Note that Jupyter Notebook support unavailable
    at this time.

    Args:
        api_url (str): The base URL for the Kumo API
        (e.g., 'https://kumorfm.ai').

    Raises:
        ImportError: If not running in a Google Colab environment or
        required modules are missing.
    """
    try:
        from google.colab import output
        from IPython.display import HTML, display
    except Exception:
        raise ImportError(
            'This method is meant to be used in Google Colab.\n If your' +
            'python code is running on your local machine, use ' +
            'kumo.authenticate_local().\n Otherwise, visit ' +
            f'{api_url}/api-keys to generate an API key.')
    else:
        import uuid
        from datetime import datetime

        token_name = "sdk-colab-" + datetime.now().strftime(
            '%Y-%m-%d-%H-%M-%S') + '-Z'

        def handle_api_key(api_key: str) -> None:
            os.environ['KUMO_API_KEY'] = api_key

        callback_id = 'api-key-button-' + str(uuid.uuid4())

        output.register_callback(callback_id, handle_api_key)

        display(
            HTML(f"""
        <div style="padding: 10px;">
            <!-- <script src="https://cdn.tailwindcss.com"></script> -->
            <svg width="100" height="50" viewBox="0 0 184 92" fill="none"
                xmlns="http://www.w3.org/2000/svg">
                <g clip-path="url(#clip0_749_1962)">
                    <path d="M67.159 67.919V46.238L53.494 59.491L68.862 82.3H61.567L49.1 63.74L42.089 70.54V82.3H36.069V29.605H42.089V62.182L58.731 45.822H73.109V67.922C73.109 73.375 76.72 77.341 82.386 77.341C87.933 77.341 91.526 73.441 91.586 68.059V0H0V91.586H91.586V80.317C88.891 82.1996 85.6731 83.1888 82.386 83.145C73.817 83.145 67.16 77.125 67.16 67.919H67.159Z" # noqa: E501
                        fill="#FC1373"/>
                    <path d="M147.398 82.296H141.379V58.715C141.379 53.97 138.262 50.429 133.447 50.429C128.632 50.429 125.515 53.97 125.515 58.715V82.298H119.495V58.715C119.495 53.97 116.379 50.429 111.563 50.429C106.747 50.429 103.631 53.97 103.631 58.715V82.298H97.611V58.925C97.611 50.993 103.347 44.902 111.563 44.902C113.756 44.8229 115.929 45.3412 117.85 46.4016C119.771 47.4619 121.367 49.0244 122.469 50.922C123.592 49.0276 125.204 47.4696 127.135 46.4107C129.066 45.3517 131.246 44.8307 133.447 44.902C141.732 44.902 147.398 50.993 147.398 58.925V82.296Z"
                        fill="#FC1373"/>
                    <path d="M183.909 64.166C183.909 74.647 175.977 83.217 165.567 83.217C155.226 83.217 147.224 74.648 147.224 64.166C147.224 53.543 155.224 44.903 165.567 44.903C175.976 44.903 183.909 53.543 183.909 64.166ZM177.889 64.166C177.889 56.73 172.366 50.639 165.567 50.639C158.839 50.639 153.315 56.73 153.315 64.166C153.315 71.461 158.839 77.41 165.567 77.41C172.367 77.41 177.889 71.461 177.889 64.166Z"
                        fill="#FC1373"/>
                </g>
                <defs>
                    <clipPath id="clip0_749_1962">
                        <rect width="183.908" height="91.586" fill="white"/>
                    </clipPath>
                </defs>
            </svg>
            <div id="prompt">
                <p>
                    Click the button below to connect to KumoRFM and
                    generate your API key.
                </p>
                <button id="{callback_id}">
                    Generate API Key
                </button>
            </div>
            <div id="success" style="display: none;">
                <p>
                    ✓ Your API key has been created and configured in your
                    colab notebook.
                </p>
                To manage all your API keys, visit the
                <a href="{api_url}/api-keys" target="_blank">
                    KumoRFM website.
                </a>
            </div>
            <div id="failed" style="display: none; color: red;">
                <p>
                    API key creation failed with error:
                    <span id="error-message"></span>
                </p>
            </div>
            <script>
                // Listen for messages from the popup
                window.addEventListener('message', function(event) {{
                    if (event.data.type === 'API_KEY_GENERATED') {{
                        // Call the Python callback with the API key
                        google.colab.kernel.invokeFunction(
                            '{callback_id}', [event.data.apiKey], {{}}
                        );
                        document.getElementById('prompt')
                            .style.display = "none";
                        document.getElementById('success')
                            .style.display = "block";
                        document.getElementById('failed')
                            .style.display = "none";
                    }} else if (
                        event.data.type === 'API_KEY_GENERATION_FAILED'
                    ) {{
                        document.getElementById('failed')
                            .style.display = "block";
                        document.getElementById('error-message')
                            .innerHTML = event.data.errorMessage;
                    }}
                }});

                document.getElementById('{callback_id}')
                    .onclick = function() {{
                        // Open the popup
                        const popup = window.open(
                            '{api_url}/authenticate-sdk?opener=colab&token_name={token_name}',
                            'apiKeyPopup',
                            'width=600,height=700,scrollbars=yes,resizable=yes'
                        );

                        // Focus the popup
                        if (popup) {{
                            popup.focus();
                        }}
                    }};
            </script>
        </div>
        """))
