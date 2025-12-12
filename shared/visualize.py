#!/usr/bin/env python3
"""
Pluggable Visualization Backend System.

Provides a unified interface for plotting that works in both terminal (plotext)
and graphical (matplotlib) environments. Use plotext for quick terminal-based
diagnostics during training, matplotlib for publication-quality figures.

Backend Selection
-----------------
- **plotext**: Terminal plotting, no GUI required, fast, low memory
- **matplotlib**: File/GUI output, publication quality, supports subplots

Usage Example
-------------
    from shared.visualize import get_backend
    
    # Terminal plotting (default)
    backend = get_backend("plotext")
    backend.plot_line(x, y, title="Loss Curve")
    backend.show()
    
    # File output
    backend = get_backend("matplotlib")
    backend.plot_scatter(actual, predicted, title="Predictions")
    backend.save("output/predictions.png")
    
    # Auto-select best available
    backend = get_backend()  # plotext if available, else matplotlib

Pluggable Design
----------------
New backends can be added by implementing the VisualizationBackend protocol:
    
    class MyBackend:
        def plot_line(self, x, y, *, label=None, title=None, xlabel=None, ylabel=None): ...
        def plot_scatter(self, x, y, *, label=None, title=None, xlabel=None, ylabel=None): ...
        def plot_histogram(self, data, bins=50, *, title=None, xlabel=None, ylabel=None): ...
        def plot_bar(self, labels, values, *, title=None, xlabel=None, ylabel=None): ...
        def show(self): ...
        def save(self, path: str): ...
        def clear(self): ...
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Union, Sequence, Any
import warnings

import numpy as np

# Type alias for array-like data
ArrayLike = Union[List[float], np.ndarray, "torch.Tensor"]


def _to_numpy(data: ArrayLike) -> np.ndarray:
    """Convert array-like data to numpy array."""
    if hasattr(data, "detach"):  # torch.Tensor
        return data.detach().cpu().numpy()
    elif hasattr(data, "numpy"):  # tf.Tensor or similar
        return data.numpy()
    else:
        return np.asarray(data)


# =============================================================================
# BACKEND PROTOCOL
# =============================================================================

class VisualizationBackend(ABC):
    """
    Abstract base class defining the visualization backend interface.
    
    All backends must implement these methods to ensure consistent API
    across terminal and graphical environments.
    """
    
    @abstractmethod
    def plot_line(
        self,
        y: ArrayLike,
        x: Optional[ArrayLike] = None,
        *,
        label: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        """Plot a line chart."""
        pass
    
    @abstractmethod
    def plot_scatter(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        label: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        """Plot a scatter chart."""
        pass
    
    @abstractmethod
    def plot_histogram(
        self,
        data: ArrayLike,
        bins: int = 50,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        """Plot a histogram."""
        pass
    
    @abstractmethod
    def plot_bar(
        self,
        labels: Sequence[str],
        values: ArrayLike,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
        horizontal: bool = False,
    ) -> None:
        """Plot a bar chart."""
        pass
    
    @abstractmethod
    def plot_hline(
        self,
        y: float,
        *,
        label: Optional[str] = None,
        color: Optional[str] = None,
        linestyle: Optional[str] = None,
    ) -> None:
        """Plot a horizontal reference line."""
        pass
    
    @abstractmethod
    def plot_diagonal(
        self,
        *,
        label: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        """Plot a diagonal y=x reference line (for scatter plots)."""
        pass
    
    @abstractmethod
    def show(self) -> None:
        """Display the current plot."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the current plot to a file."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear the current plot/figure."""
        pass
    
    @abstractmethod
    def set_title(self, title: str) -> None:
        """Set the plot title."""
        pass
    
    @abstractmethod
    def set_labels(self, xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> None:
        """Set axis labels."""
        pass
    
    @abstractmethod
    def legend(self) -> None:
        """Show legend."""
        pass


# =============================================================================
# PLOTEXT BACKEND (Terminal)
# =============================================================================

class PlotextBackend(VisualizationBackend):
    """
    Terminal-based plotting using plotext.
    
    Great for:
    - Quick diagnostics during training
    - SSH sessions without X forwarding
    - CI/CD environments
    - Low-memory environments
    
    Limitations:
    - Resolution limited by terminal size
    - No subplot support (plots sequentially)
    - Limited color/style options
    """
    
    def __init__(self):
        try:
            import plotext as plt
            self._plt = plt
        except ImportError:
            raise ImportError(
                "plotext is required for terminal plotting.\n"
                "Install with: pip install plotext"
            )
        self._has_data = False
    
    def plot_line(
        self,
        y: ArrayLike,
        x: Optional[ArrayLike] = None,
        *,
        label: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        y_np = _to_numpy(y).flatten()
        if x is not None:
            x_np = _to_numpy(x).flatten()
        else:
            x_np = np.arange(len(y_np))
        
        kwargs = {}
        if label:
            kwargs["label"] = label
        if color:
            kwargs["color"] = color
        
        self._plt.plot(x_np.tolist(), y_np.tolist(), **kwargs)
        self._has_data = True
        
        if title:
            self._plt.title(title)
        if xlabel:
            self._plt.xlabel(xlabel)
        if ylabel:
            self._plt.ylabel(ylabel)
    
    def plot_scatter(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        label: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        x_np = _to_numpy(x).flatten()
        y_np = _to_numpy(y).flatten()
        
        kwargs = {}
        if label:
            kwargs["label"] = label
        if color:
            kwargs["color"] = color
        
        self._plt.scatter(x_np.tolist(), y_np.tolist(), **kwargs)
        self._has_data = True
        
        if title:
            self._plt.title(title)
        if xlabel:
            self._plt.xlabel(xlabel)
        if ylabel:
            self._plt.ylabel(ylabel)
    
    def plot_histogram(
        self,
        data: ArrayLike,
        bins: int = 50,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        data_np = _to_numpy(data).flatten()
        
        kwargs = {"bins": bins}
        if color:
            kwargs["color"] = color
        
        self._plt.hist(data_np.tolist(), **kwargs)
        self._has_data = True
        
        if title:
            self._plt.title(title)
        if xlabel:
            self._plt.xlabel(xlabel)
        if ylabel:
            self._plt.ylabel(ylabel or "Count")
    
    def plot_bar(
        self,
        labels: Sequence[str],
        values: ArrayLike,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
        horizontal: bool = False,
    ) -> None:
        values_np = _to_numpy(values).flatten()
        
        kwargs = {}
        if color:
            kwargs["color"] = color
        
        if horizontal:
            self._plt.bar(list(labels), values_np.tolist(), orientation="horizontal", **kwargs)
        else:
            self._plt.bar(list(labels), values_np.tolist(), **kwargs)
        self._has_data = True
        
        if title:
            self._plt.title(title)
        if xlabel:
            self._plt.xlabel(xlabel)
        if ylabel:
            self._plt.ylabel(ylabel)
    
    def plot_hline(
        self,
        y: float,
        *,
        label: Optional[str] = None,
        color: Optional[str] = None,
        linestyle: Optional[str] = None,
    ) -> None:
        self._plt.hline(y, color=color or "red")
        # Note: plotext hline doesn't support labels directly
    
    def plot_diagonal(
        self,
        *,
        label: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        # Get current plot limits and draw y=x line
        # Note: plotext doesn't have easy axis limit access, so we approximate
        self._plt.plot([0, 1], [0, 1], color=color or "gray")
    
    def show(self) -> None:
        if self._has_data:
            self._plt.show()
            self._has_data = False
    
    def save(self, path: str) -> None:
        # plotext can save to file
        self._plt.savefig(path)
        self._has_data = False
    
    def clear(self) -> None:
        self._plt.clear_figure()
        self._has_data = False
    
    def set_title(self, title: str) -> None:
        self._plt.title(title)
    
    def set_labels(self, xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> None:
        if xlabel:
            self._plt.xlabel(xlabel)
        if ylabel:
            self._plt.ylabel(ylabel)
    
    def legend(self) -> None:
        # plotext shows legend automatically with labels
        pass


# =============================================================================
# MATPLOTLIB BACKEND (File/GUI)
# =============================================================================

class MatplotlibBackend(VisualizationBackend):
    """
    Graphical plotting using matplotlib.
    
    Great for:
    - Publication-quality figures
    - Complex layouts with subplots
    - Saving to various formats (PNG, PDF, SVG)
    - Interactive exploration
    
    Requirements:
    - matplotlib installed
    - Display available (or use Agg backend for headless)
    """
    
    def __init__(self, figsize: tuple = (10, 6), style: str = "seaborn-v0_8-whitegrid"):
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend by default
            import matplotlib.pyplot as plt
            self._plt = plt
            self._matplotlib = matplotlib
        except ImportError:
            raise ImportError(
                "matplotlib is required for graphical plotting.\n"
                "Install with: pip install matplotlib"
            )
        
        # Try to set style, fall back gracefully
        try:
            self._plt.style.use(style)
        except Exception:
            pass  # Style not available, use default
        
        self._figsize = figsize
        self._fig = None
        self._ax = None
        self._ensure_figure()
    
    def _ensure_figure(self):
        """Create figure/axes if not exists."""
        if self._fig is None or self._ax is None:
            self._fig, self._ax = self._plt.subplots(figsize=self._figsize)
    
    def plot_line(
        self,
        y: ArrayLike,
        x: Optional[ArrayLike] = None,
        *,
        label: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        self._ensure_figure()
        y_np = _to_numpy(y).flatten()
        if x is not None:
            x_np = _to_numpy(x).flatten()
        else:
            x_np = np.arange(len(y_np))
        
        kwargs = {}
        if label:
            kwargs["label"] = label
        if color:
            kwargs["color"] = color
        
        self._ax.plot(x_np, y_np, **kwargs)
        
        if title:
            self._ax.set_title(title)
        if xlabel:
            self._ax.set_xlabel(xlabel)
        if ylabel:
            self._ax.set_ylabel(ylabel)
    
    def plot_scatter(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        label: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
        alpha: float = 0.5,
    ) -> None:
        self._ensure_figure()
        x_np = _to_numpy(x).flatten()
        y_np = _to_numpy(y).flatten()
        
        kwargs = {"alpha": alpha, "s": 10}
        if label:
            kwargs["label"] = label
        if color:
            kwargs["color"] = color
        
        self._ax.scatter(x_np, y_np, **kwargs)
        
        if title:
            self._ax.set_title(title)
        if xlabel:
            self._ax.set_xlabel(xlabel)
        if ylabel:
            self._ax.set_ylabel(ylabel)
    
    def plot_histogram(
        self,
        data: ArrayLike,
        bins: int = 50,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        self._ensure_figure()
        data_np = _to_numpy(data).flatten()
        
        kwargs = {"bins": bins, "alpha": 0.7, "edgecolor": "black"}
        if color:
            kwargs["color"] = color
        
        self._ax.hist(data_np, **kwargs)
        
        if title:
            self._ax.set_title(title)
        if xlabel:
            self._ax.set_xlabel(xlabel)
        if ylabel:
            self._ax.set_ylabel(ylabel or "Count")
    
    def plot_bar(
        self,
        labels: Sequence[str],
        values: ArrayLike,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
        horizontal: bool = False,
    ) -> None:
        self._ensure_figure()
        values_np = _to_numpy(values).flatten()
        
        kwargs = {"alpha": 0.7}
        if color:
            kwargs["color"] = color
        
        if horizontal:
            self._ax.barh(list(labels), values_np, **kwargs)
        else:
            self._ax.bar(list(labels), values_np, **kwargs)
        
        if title:
            self._ax.set_title(title)
        if xlabel:
            self._ax.set_xlabel(xlabel)
        if ylabel:
            self._ax.set_ylabel(ylabel)
    
    def plot_hline(
        self,
        y: float,
        *,
        label: Optional[str] = None,
        color: Optional[str] = None,
        linestyle: Optional[str] = None,
    ) -> None:
        self._ensure_figure()
        kwargs = {}
        if label:
            kwargs["label"] = label
        if color:
            kwargs["color"] = color
        if linestyle:
            kwargs["linestyle"] = linestyle
        
        self._ax.axhline(y, **kwargs)
    
    def plot_diagonal(
        self,
        *,
        label: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        self._ensure_figure()
        # Get current axis limits
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        
        # Draw diagonal from min to max
        min_val = min(xlim[0], ylim[0])
        max_val = max(xlim[1], ylim[1])
        
        kwargs = {"linestyle": "--", "alpha": 0.7}
        if label:
            kwargs["label"] = label
        if color:
            kwargs["color"] = color
        else:
            kwargs["color"] = "gray"
        
        self._ax.plot([min_val, max_val], [min_val, max_val], **kwargs)
    
    def show(self) -> None:
        if self._fig is not None:
            self._plt.tight_layout()
            self._plt.show()
            self._fig = None
            self._ax = None
    
    def save(self, path: str) -> None:
        if self._fig is not None:
            self._plt.tight_layout()
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  ✓ Saved plot to: {path}")
    
    def clear(self) -> None:
        if self._fig is not None:
            self._plt.close(self._fig)
        self._fig = None
        self._ax = None
    
    def set_title(self, title: str) -> None:
        self._ensure_figure()
        self._ax.set_title(title)
    
    def set_labels(self, xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> None:
        self._ensure_figure()
        if xlabel:
            self._ax.set_xlabel(xlabel)
        if ylabel:
            self._ax.set_ylabel(ylabel)
    
    def legend(self) -> None:
        self._ensure_figure()
        self._ax.legend()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

# Track available backends
_BACKENDS = {
    "plotext": PlotextBackend,
    "matplotlib": MatplotlibBackend,
}

_BACKEND_AVAILABLE = {}


def _check_backend_available(name: str) -> bool:
    """Check if a backend is available (cached)."""
    if name not in _BACKEND_AVAILABLE:
        try:
            _BACKENDS[name]()
            _BACKEND_AVAILABLE[name] = True
        except ImportError:
            _BACKEND_AVAILABLE[name] = False
    return _BACKEND_AVAILABLE[name]


def get_backend(name: Optional[str] = None) -> VisualizationBackend:
    """
    Get a visualization backend by name.
    
    Args:
        name: Backend name ("plotext", "matplotlib") or None for auto-detect.
              Auto-detect prefers plotext (terminal-friendly).
    
    Returns:
        VisualizationBackend instance.
    
    Raises:
        ImportError: If requested backend is not available.
        ValueError: If backend name is unknown.
    
    Example:
        >>> backend = get_backend()  # Auto-detect
        >>> backend = get_backend("plotext")  # Terminal
        >>> backend = get_backend("matplotlib")  # File/GUI
    """
    if name is None:
        # Auto-detect: prefer plotext for terminal
        for backend_name in ["plotext", "matplotlib"]:
            if _check_backend_available(backend_name):
                return _BACKENDS[backend_name]()
        raise ImportError(
            "No visualization backend available.\n"
            "Install plotext: pip install plotext\n"
            "Or matplotlib: pip install matplotlib"
        )
    
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}")
    
    return _BACKENDS[name]()


def list_backends() -> dict:
    """
    List all backends and their availability.
    
    Returns:
        Dict mapping backend name to availability (bool).
    
    Example:
        >>> list_backends()
        {'plotext': True, 'matplotlib': True}
    """
    return {name: _check_backend_available(name) for name in _BACKENDS}
