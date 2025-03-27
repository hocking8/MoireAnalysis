from typing import Any, Dict, Tuple

import numpy as np
import pySPM

from .classes import Channel, Scan


def read_bruker(file_path: str) -> Scan:
    def format_all_metadata(
        raw_channel_meta_data,
        raw_scan_meta_data,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            channel_name: extract_metadata(
                raw_channel_meta_data,
                raw_scan_meta_data,
                channel_name,
            )
            for channel_name in list(raw_channel_meta_data.keys())
        }

    def read_metadata(file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        lines = []
        with open(file_path, "r", encoding="ISO-8859-1") as f:
            for line in f:
                line = line.strip()
                if r"\*File list end" in line:
                    break
                lines.append(line)

        meta_list = []
        current_context = []
        for line in lines:
            line = line[1:]  # remove the first two \

            if line.startswith("*"):
                if len(current_context) > 0:
                    meta_list.append(current_context)
                    current_context = []
                current_context.append(line[1:])
            else:
                kv = line.split(": ")
                if len(kv) == 2:
                    key, value = kv
                else:
                    key = kv[0]
                    value = ""
                current_context.append((key, value))

        meta_list.append(current_context)

        # now transform the list of lists into a dictionary
        scan_meta_data = {}
        channel_meta_data = {}
        for i, context in enumerate(meta_list):
            if context[0] != "Ciao image list":
                scan_meta_data[context[0]] = dict(context[1:])
            else:
                temp_dict = dict(context[1:])
                channel_name = temp_dict["@2:Image Data"]
                channel_name = channel_name.split('"')[1]

                trace_direction = temp_dict["Line Direction"]
                channel_name = channel_name + ": " + trace_direction
                channel_meta_data[channel_name] = temp_dict

        return scan_meta_data, channel_meta_data

    def extract_channel_data(bruker_scan, channel_name: str) -> np.ndarray:
        channel, direction = channel_name.split(": ")
        if direction == "Trace":
            backward = False
        elif direction == "Retrace":
            backward = True
        else:
            raise ValueError(f"Invalid direction, {direction} is not a valid direction")

        channel = bruker_scan.get_channel(channel, backward=backward)

        return channel.pixels

    def format_metadata(bruker_scan, raw_channel_meta_data) -> np.ndarray:
        return {
            channel_name: extract_channel_data(bruker_scan, channel_name)
            for channel_name in list(raw_channel_meta_data.keys())
        }

    def extract_metadata(
        raw_channel_meta_data, raw_scan_meta_data, channel_name: str
    ) -> Dict[str, Any]:
        """
        Returns the metadata for a given channel

        The metadata includes the following keys:

        | Key             | Value Type                | .spm metakey                         |
        | --------------- | ------------------------- | ------------------------------------ |
        | scan_direction  | Up/Down                   | Frame direction                      |
        | trace_direction | Trace/Retrace             | Line Direction                       |
        | scan_size       | Tuple of (X, Y) in meters | Scan Size                            |
        | scan_rounding   | float                     | X Round                              |
        | scan_offset     | Tuple of (X, Y) in meters | X Offset / Y Offset [Ciao scan list] |
        | scan_angle      | float in radians          | Rotate Ang. [Ciao scan list]         |
        | scan_rate       | float in hertz            | Scan Rate [Ciao scan list]           |

        ### Additional metadata not yet added
        Drive Amplitude   | @2:TR Drive Amplitude [Ciao scan list]
        Loading Force     | @2:SetTRVertDefl [Ciao scan list]
        """

        current_channel_meta_data = raw_channel_meta_data[channel_name]
        scan_list = raw_scan_meta_data["Ciao scan list"]

        scan_rounding = float(scan_list["X round"])
        scan_offset_x, scan_offset_x_unit = scan_list["X Offset"].split(" ")
        scan_offset_y, scan_offset_y_unit = scan_list["Y Offset"].split(" ")
        scan_size_x, scan_size_y, scan_size_unit = current_channel_meta_data[
            "Scan Size"
        ].split(" ")

        # convert all units to meters
        if scan_offset_x_unit == "um":
            scan_offset_x = float(scan_offset_x) * 1e3
        elif scan_offset_x_unit == "nm":
            scan_offset_x = float(scan_offset_x)
        else:
            raise ValueError(
                f"Invalid unit {scan_offset_x_unit}, valid units are um, nm"
            )

        if scan_offset_y_unit == "um":
            scan_offset_y = float(scan_offset_y) * 1e3
        elif scan_offset_y_unit == "nm":
            scan_offset_y = float(scan_offset_y)
        else:
            raise ValueError(
                f"Invalid unit {scan_offset_y_unit}, valid units are um, nm"
            )

        if scan_size_unit == "um" or scan_size_unit == "~m":
            scan_size_x = float(scan_size_x) * 1e3
            scan_size_y = float(scan_size_y) * 1e3
        elif scan_size_unit == "nm":
            scan_size_x = float(scan_size_x)
            scan_size_y = float(scan_size_y)
        else:
            raise ValueError(f"Invalid unit {scan_size_unit}, valid units are um, nm")

        return {
            "scan_direction": current_channel_meta_data["Frame direction"],
            "trace_direction": current_channel_meta_data["Line Direction"],
            "scan_size": (scan_size_x, scan_size_y),
            "scan_offset": (scan_offset_x, scan_offset_y),
            "scan_angle": float(scan_list["Rotate Ang."]),
            "scan_rate": float(scan_list["Scan Rate"]),
            "scan_rounding": scan_rounding,
        }

    bruker_scan = pySPM.Bruker(file_path)
    raw_scan_meta_data, raw_channel_meta_data = read_metadata(file_path)

    meta_data = format_all_metadata(raw_channel_meta_data, raw_scan_meta_data)
    channel_data = format_metadata(bruker_scan, raw_channel_meta_data)

    channels = {}
    for channel_name, channel_data in channel_data.items():
        channels[channel_name] = Channel(
            name=channel_name,
            data=channel_data,
            scan_direction=meta_data[channel_name]["scan_direction"],
            trace_direction=meta_data[channel_name]["trace_direction"],
            scan_size=meta_data[channel_name]["scan_size"],
        )

    return Scan(
        name=file_path,
        channels=channels,
        scan_offset=meta_data[channel_name]["scan_offset"],
        scan_angle=meta_data[channel_name]["scan_angle"],
        scan_rate=meta_data[channel_name]["scan_rate"],
        scan_rounding=meta_data[channel_name]["scan_rounding"],
    )
