import os
import pickle
from typing import Dict, List, Tuple

from .channel import Channel


class Scan:
    def __init__(
        self,
        name: str | None = None,
        channels: Dict[str, Channel] | None = None,
        scan_offset: Tuple[float, float] | None = None,
        scan_angle: float | None = None,
        scan_rate: float | None = None,
        scan_rounding: float | None = None,
        tip_velocity: Dict[str, float] = {},
    ) -> None:
        self.name = name
        self.channels = channels
        self.scan_offset = scan_offset
        self.scan_angle = scan_angle
        self.scan_rate = scan_rate
        self.scan_rounding = scan_rounding

        # tip velocity is a dictionary with the keys "fast_retrace", "slow_retrace", "fast_trace", "slow_trace"
        self.tip_velocity = tip_velocity

    def list_channels(self) -> List[str]:
        return list(self.channels.keys())

    def __repr__(self):
        return f"Scan(name={self.name}, channels={self.channels})"

    def __copy__(self):
        return Scan(
            name=self.name,
            channels={key: value.__copy__() for key, value in self.channels.items()},
            scan_offset=self.scan_offset,
            scan_angle=self.scan_angle,
            scan_rate=self.scan_rate,
            tip_velocity={
                key: value.copy() for key, value in self.tip_velocity.items()
            },
            scan_rounding=self.scan_rounding,
        )

    def save(self, filename: str, overwrite: bool = False) -> None:
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(
                f"File {filename} already exists. Use overwrite=True to overwrite it."
            )
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(filename: str) -> "Scan":
        with open(filename, "rb") as f:
            return pickle.load(f)
