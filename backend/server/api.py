"""
Flask REST API Server
======================
Exposes the live simulation state to the frontend via simple GET
endpoints.  The frontend polls these endpoints (e.g. every 100 ms)
using ``fetch()``.

Endpoints
---------
    GET /state    → current intersection state (lights, queues, etc.)
    GET /metrics  → aggregated performance metrics
    GET /config   → controller type and parameters
    GET /history  → GA evolution history (only when using GA mode)
"""

from flask import Flask, jsonify, Blueprint

api_blueprint = Blueprint("api", __name__)

# These will be set by main.py before the server starts
_shared_state: dict = {}
_state_lock = None
_metrics_collector = None
_controller = None
_controller_mode: str = "fixed"


def init_api(
    shared_state: dict,
    state_lock,
    metrics_collector,
    controller,
    controller_mode: str,
) -> None:
    """
    Inject references to the shared simulation objects.
    Called once from main.py before starting the Flask app.
    """
    global _shared_state, _state_lock, _metrics_collector
    global _controller, _controller_mode
    _shared_state = shared_state
    _state_lock = state_lock
    _metrics_collector = metrics_collector
    _controller = controller
    _controller_mode = controller_mode


@api_blueprint.route("/state", methods=["GET"])
def get_state():
    """Current intersection state — polled by the frontend."""
    with _state_lock:
        return jsonify(_shared_state)


@api_blueprint.route("/metrics", methods=["GET"])
def get_metrics():
    """Aggregated performance summary."""
    if _metrics_collector is None:
        return jsonify({"error": "Metrics collector not initialised"}), 500
    return jsonify(_metrics_collector.summary())


@api_blueprint.route("/metrics/queues", methods=["GET"])
def get_queue_history():
    """Queue lengths over time (for live graphs)."""
    if _metrics_collector is None:
        return jsonify({"error": "Metrics collector not initialised"}), 500
    # Return last 600 ticks (~ 60 seconds at 10 ticks/s)
    data = _metrics_collector.queue_length_over_time()
    return jsonify(data[-600:])


@api_blueprint.route("/config", methods=["GET"])
def get_config():
    """Current controller configuration."""
    info = {
        "controller_mode": _controller_mode,
        "controller": str(_controller),
    }
    if _controller_mode == "ga":
        ns, ew = _controller.get_current_timings()
        info["current_timings"] = {"ns_green": ns, "ew_green": ew}
    elif _controller_mode == "fixed":
        ns, ew = _controller.get_current_timings()
        info["current_timings"] = {"ns_green": ns, "ew_green": ew}
    return jsonify(info)


@api_blueprint.route("/history", methods=["GET"])
def get_evolution_history():
    """GA evolution history (returns empty list for fixed-time mode)."""
    if hasattr(_controller, "get_evolution_history"):
        return jsonify(_controller.get_evolution_history())
    return jsonify([])


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Enable CORS for local frontend development
    try:
        from flask_cors import CORS
        CORS(app)
    except ImportError:
        print("WARNING: flask-cors not installed. Frontend may have CORS issues.")

    app.register_blueprint(api_blueprint)
    return app
