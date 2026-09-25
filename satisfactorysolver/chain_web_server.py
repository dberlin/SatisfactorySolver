"""
Serve the chain visualizer with live re-solving, so recipes can be switched on
and off (and targets edited) from the browser.
"""

import gc
import json
import logging
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from satisfactorysolver.web_visualizer import capture_chain_solution, render_html

logger = logging.getLogger(__name__)


def format_rates(rates: dict[str, Fraction]) -> str:
    """Format rates as the ITEM=RATE lines the web UI edits."""
    return "\n".join(f"{item}={rate}" for item, rate in rates.items())


class ChainSession:
    """Solves chain requests from the web UI against one set of game data."""

    def __init__(self, recipes, item_names, finder_class, parse_targets):
        """
        :param recipes: Every recipe the user may enable.
        :param item_names: Valid item names.
        :param finder_class: OptimalChainFinder subclass to solve with.
        :param parse_targets: Callable (input lines, output lines, item names) ->
            (inputs, outputs) raising ValueError on bad targets.
        """
        self.recipes = set(recipes)
        self.recipe_names = {recipe.Name for recipe in self.recipes}
        self.item_names = item_names
        self.finder_class = finder_class
        self.parse_targets = parse_targets
        # z3's default context is not thread-safe, and that includes freeing z3
        # objects, which Python's garbage collector may do on whichever thread
        # happens to trigger it. So every solver object lives and dies on this
        # one thread; a lock alone let another request's thread free them
        # mid-solve and crash the process.
        self.solver_thread = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="chain-solver"
        )

    def _solve_on_solver_thread(self, enabled, inputs, outputs):
        """Solve and return only plain data, so no solver object leaves this thread."""
        try:
            candidates = self.finder_class(
                self.recipes
            ).construct_possibly_used_recipes(outputs)
            finder = self.finder_class(enabled)
            finder.build_model(inputs, outputs)
            solution = (
                capture_chain_solution(finder, "Chain") if finder.solve() else None
            )
            del finder
            return candidates, solution
        finally:
            # Collect this solve's garbage here rather than on another thread.
            gc.collect()

    def solve(self, request: dict) -> dict:
        disabled = set(request.get("disabled", [])) & self.recipe_names
        try:
            inputs, outputs = self.parse_targets(
                _lines(request.get("inputs", "")),
                _lines(request.get("outputs", "")),
                self.item_names,
            )
            if not outputs:
                raise ValueError("enter at least one output")
        except ValueError as error:
            return {"error": str(error), "solution": None, "recipes": None}

        enabled = {recipe for recipe in self.recipes if recipe.Name not in disabled}
        started = time.perf_counter()
        candidates, solution = self.solver_thread.submit(
            self._solve_on_solver_thread, enabled, inputs, outputs
        ).result()
        elapsed = time.perf_counter() - started
        solved = solution is not None

        used = {}
        if solution:
            used = {
                node["recipe"]: node for node in solution["nodes"] if node.get("recipe")
            }
        recipes = [
            {
                "name": recipe.Name,
                "machine": recipe.Machine.Name,
                "alternate": bool(recipe.Alternate),
                "products": [part.Name for part, _ in recipe.Outputs],
                "ingredients": [part.Name for part, _ in recipe.Inputs],
                "enabled": recipe.Name not in disabled,
                "used": recipe.Name in used,
                "buildings": used[recipe.Name]["buildings"]
                if recipe.Name in used
                else None,
            }
            for recipe in sorted(candidates, key=lambda r: r.Name)
        ]
        return {
            "error": None
            if solved
            else "No optimal production chain found (infeasible, unbounded, or solver unknown).",
            "solution": solution,
            "recipes": recipes,
            "seconds": elapsed,
        }


def _lines(text) -> list[str]:
    if isinstance(text, list):
        return [str(line) for line in text]
    return [line for line in str(text).splitlines() if line.strip()]


def make_handler(session: ChainSession, page: str):
    class ChainRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/index.html"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", page.encode())

        def do_POST(self):
            if self.path != "/api/solve":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length) or b"{}")
            except ValueError as error:
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"error": f"bad request: {error}"}
                )
                return
            if not isinstance(request, dict):
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"error": "bad request: expected an object"}
                )
                return
            try:
                response = session.solve(request)
            except Exception as error:
                logger.exception("Solve failed")
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
                return
            self._send_json(HTTPStatus.OK, response)

        def _send_json(self, status, body):
            self._send(status, "application/json", json.dumps(body).encode())

        def _send(self, status, content_type, body: bytes):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            logger.debug("%s - %s", self.address_string(), format % args)

    return ChainRequestHandler


def serve(
    session: ChainSession,
    inputs: dict[str, Fraction],
    outputs: dict[str, Fraction],
    disabled: set[str],
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = False,
) -> None:
    """Serve the live chain visualizer until interrupted."""
    page = render_html(
        [],
        "Production chain",
        live={
            "inputs": format_rates(inputs),
            "outputs": format_rates(outputs),
            "disabled": sorted(disabled),
        },
    )
    server = ThreadingHTTPServer((host, port), make_handler(session, page))
    url = f"http://{host}:{server.server_port}/"
    print(f"Serving the chain visualizer at {url} (Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
