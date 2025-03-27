from typing import List, Literal, Tuple

import numpy as np


class Channel:
    def __init__(
        self,
        name: str | None = None,
        data: np.ndarray | None = None,
        scan_direction: Literal["Up", "Down"] | None = None,
        trace_direction: Literal["Trace", "Retrace"] | None = None,
        scan_size: Tuple[float, float] | None = None,
        scan_size_std: Tuple[float, float] = (0, 0),
        peak_positions: List[Tuple[float, float]] = [],
        peak_positions_std: List[Tuple[float, float]] = [],
    ) -> None:
        self.name = name
        self.data = data
        self.scan_direction = scan_direction
        self.trace_direction = trace_direction
        self.scan_size = scan_size
        self.scan_size_std = scan_size_std
        self.peak_positions = peak_positions
        self.peak_positions_std = peak_positions_std

    def __repr__(self):
        return f"Channel(name={self.name})"

    def __copy__(self):
        return Channel(
            self.name,
            self.data.copy() if self.data is not None else None,
            self.scan_direction,
            self.trace_direction,
            self.scan_size,
            self.scan_size_std,
            self.peak_positions.copy(),
            self.peak_positions_std.copy(),
        )
