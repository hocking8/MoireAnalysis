from typing import List, Literal, Tuple

import numpy as np
import scipy.ndimage
import scipy.signal
import skimage.feature
import uncertainties as u
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import curve_fit

from .classes import Scan


def correct_background(
    scan: Scan,
    method: Literal["mean", "plane"] = "plane",
    channel_names: List[str] | None = None,
    inplace: bool = False,
) -> Scan:
    if channel_names is None:
        channel_names = [
            channel_name
            for channel_name in scan.list_channels()
            if not (
                channel_name.startswith("Xsensor") or channel_name.startswith("Ysensor")
            )
        ]

    if not inplace:
        corrected_scan = scan.__copy__()
    else:
        corrected_scan = scan

    for channel_name in channel_names:
        data = corrected_scan.channels[channel_name].data

        if method == "mean":
            background = data.mean()
        elif method == "plane":
            x = np.arange(data.shape[1])
            y = np.arange(data.shape[0])
            X, Y = np.meshgrid(x, y)
            A = np.column_stack((np.ones(data.ravel().size), X.ravel(), Y.ravel()))
            c, _, _, _ = np.linalg.lstsq(A, data.ravel(), rcond=-1)
            background = c[0] * np.ones(data.shape) + c[1] * X + c[2] * Y
        else:
            raise ValueError(f"Invalid method: {method}")

        corrected_scan.channels[channel_name].data = data - background

    return corrected_scan


def correct_sensor(
    scan: Scan,
    polynomial_degree: int = 1,
    channel_names: List[str] | None = None,
    inplace: bool = False,
) -> Scan:
    def generate_2D_polynomial(degree: int):
        def polynomial(xy, *coefficients) -> np.ndarray:
            x, y = xy
            idx = 0
            z = np.zeros_like(x)
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    z += coefficients[idx] * (x**i) * (y**j)
                    idx += 1
            return z

        return polynomial

    if channel_names is None:
        channel_names = [
            channel_name
            for channel_name in scan.list_channels()
            if channel_name.startswith("Xsensor") or channel_name.startswith("Ysensor")
        ]

    if not inplace:
        corrected_scan = scan.__copy__()
    else:
        corrected_scan = scan

    for channel_name in channel_names:
        data = corrected_scan.channels[channel_name].data
        scan_size = corrected_scan.channels[channel_name].scan_size

        x = np.linspace(
            0,
            scan_size[0],
            data.shape[0],
        )
        y = np.linspace(
            0,
            scan_size[1],
            data.shape[1],
        )
        x, y = np.meshgrid(x, y)

        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = data.flatten()

        polynomial = generate_2D_polynomial(polynomial_degree)
        num_coefficients = (polynomial_degree + 1) * (polynomial_degree + 2) // 2
        params, _ = curve_fit(
            polynomial,
            (x_flat, y_flat),
            z_flat,
            p0=np.ones(num_coefficients),
        )
        z_fit = polynomial((x, y), *params)

        corrected_scan.channels[channel_name].data = z_fit

    return corrected_scan


def interpolate_scan(
    scan: Scan,
    inplace: bool = False,
    channel_names: List[str] | None = None,
) -> Scan:
    if not inplace:
        corrected_scan = scan.__copy__()
    else:
        corrected_scan = scan

    if channel_names is None:
        channel_names = scan.list_channels()

    for trace_direction in ["Trace", "Retrace"]:
        x_sensor = corrected_scan.channels[f"Xsensor: {trace_direction}"].data.copy()
        y_sensor = corrected_scan.channels[f"Ysensor: {trace_direction}"].data.copy()

        # extract the extremal points of the plane which the X and Y sensor data defines
        p1 = (x_sensor[0, 0], y_sensor[0, 0])
        p2 = (x_sensor[-1, 0], y_sensor[-1, 0])
        p3 = (x_sensor[0, -1], y_sensor[0, -1])
        p4 = (x_sensor[-1, -1], y_sensor[-1, -1])
        ps = [p1, p2, p3, p4]

        # have the mean of the points as the origin
        # this is done to have a consistent orientation of the points
        mean_x = np.mean([p[0] for p in ps])
        mean_y = np.mean([p[1] for p in ps])
        ps_mean = [(p[0] - mean_x, p[1] - mean_y) for p in ps]
        ps_sorted_idx = np.argsort([np.arctan2(p[1], p[0]) for p in ps_mean])
        ps_sorted = [ps[i] for i in ps_sorted_idx]

        # define a rectangle that is fully enclosed by the polygon defined by the 4 points
        x_values = [p[0] for p in ps]
        x_values.sort()
        y_values = [p[1] for p in ps]
        y_values.sort()
        # get the 2 inner points of the rectangle
        x_min, x_max = x_values[1], x_values[-2]
        y_min, y_max = y_values[1], y_values[-2]
        x_extent = x_max - x_min
        y_extent = y_max - y_min

        r1 = (x_max, y_min)
        r2 = (x_min, y_min)
        r3 = (x_max, y_max)
        r4 = (x_min, y_max)
        rs = [r1, r2, r3, r4]

        mean_x = np.mean([r[0] for r in rs])
        mean_y = np.mean([r[1] for r in rs])
        rs_mean = [(r[0] - mean_x, r[1] - mean_y) for r in rs]
        rs_sorted_idx = np.argsort([np.arctan2(r[1], r[0]) for r in rs_mean])
        rs_sorted = [rs[i] for i in rs_sorted_idx]

        # calculate the 2 vectors of the initial plane
        v1 = np.array(ps_sorted[1]) - np.array(ps_sorted[0])
        v2 = np.array(ps_sorted[-1]) - np.array(ps_sorted[0])
        v1_length = np.linalg.norm(v1)
        v2_length = np.linalg.norm(v2)

        # calculate the 2 vectors of the new rectangle
        v1_new = np.array(rs_sorted[1]) - np.array(rs_sorted[0])
        v2_new = np.array(rs_sorted[-1]) - np.array(rs_sorted[0])
        v1_new_length = np.linalg.norm(v1_new)
        v2_new_length = np.linalg.norm(v2_new)

        # calculate the length ratios of the 2 vectors
        # this is done to later interpolate the data to a regular grid by multiplying the length ratios with the original pixel size
        v1_length_ratio = v1_new_length / v1_length
        v2_length_ratio = v2_new_length / v2_length

        # interpolate the data to a regular grid
        # we multiply by the length ratio to minimize interpolation errors
        num_x = x_sensor.shape[0]  # int(x_sensor.shape[0] * v1_length_ratio)
        num_y = x_sensor.shape[1]  # int(x_sensor.shape[1] * v2_length_ratio)

        new_X = np.linspace(x_min, x_max, min(num_x, num_y))
        new_Y = np.linspace(y_min, y_max, min(num_x, num_y))
        new_X, new_Y = np.meshgrid(new_X, new_Y)

        corrected_scan.channels[f"Xsensor: {trace_direction}"].data = new_X
        corrected_scan.channels[f"Ysensor: {trace_direction}"].data = new_Y
        corrected_scan.channels[f"Xsensor: {trace_direction}"].scan_size = (
            x_extent,
            y_extent,
        )
        corrected_scan.channels[f"Ysensor: {trace_direction}"].scan_size = (
            x_extent,
            y_extent,
        )

        for channel_name in channel_names:
            # only interpolate the data of the current trace direction
            if trace_direction not in channel_name:
                continue

            X_flat = x_sensor.flatten()
            Y_flat = y_sensor.flatten()
            Z_flat = scan.channels[channel_name].data.flatten()

            interp = LinearNDInterpolator(list(zip(X_flat, Y_flat)), Z_flat)
            interpolated_data = interp(new_X, new_Y)

            corrected_scan.channels[channel_name].data = interpolated_data
            corrected_scan.channels[channel_name].scan_size = (x_extent, y_extent)

    return corrected_scan


def estimate_tip_velocity(
    scan: Scan,
    method: Literal["finite_difference", "line_fit"] = "line_fit",
    channel_names: List[str] | None = None,
    inplace: bool = False,
):
    # TODO Check if rounding is applied right
    # We need to check the bruker for that
    def line_fit_method(
        data: np.ndarray,
        axis: int,
    ) -> Tuple[float, float]:
        speeds = []
        for i in range(0, data.shape[axis]):
            linecut = data[:, i] if axis == 0 else data[i, :]
            x = np.arange(len(linecut))
            # fit a mx + b to the linecut
            # m is the speed in nm / pixel, b has no physical meaning
            popt, _ = curve_fit(lambda x, m, b: m * x + b, x, linecut)
            speeds.append(popt[0])

        v_mean = abs(np.mean(speeds))
        v_std = np.std(speeds, ddof=1)  # / np.sqrt(len(speeds))

        return v_mean, v_std

    def finite_difference_method(
        data: np.ndarray,
        axis: int,
    ) -> Tuple[float, float]:
        dx = np.gradient(data, axis=axis)
        v_mean = abs(np.mean(dx))
        v_std = np.std(dx, ddof=1) / np.sqrt(dx.size)

        return v_mean, v_std

    if channel_names is None:
        channel_names = [
            channel_name
            for channel_name in scan.list_channels()
            if channel_name.startswith("Xsensor") or channel_name.startswith("Ysensor")
        ]

    if not inplace:
        corrected_scan = scan.__copy__()
    else:
        corrected_scan = scan

    for channel_name in channel_names:
        data = corrected_scan.channels[channel_name].data
        trace_direction = corrected_scan.channels[channel_name].trace_direction
        scan_rate = corrected_scan.scan_rate
        scan_rounding = corrected_scan.scan_rounding

        # choose the axis along which the sensor is scanning
        # i.e. the axis along which the gradient is calculated
        # get the axis with the biggest gradient
        grad_axis_0 = abs(np.gradient(data, axis=0).mean())
        grad_axis_1 = abs(np.gradient(data, axis=1).mean())
        if grad_axis_0 > grad_axis_1:
            axis = 0
        else:
            axis = 1

        if method == "line_fit":
            v_mean, v_std = line_fit_method(data, axis)
        elif method == "finite_difference":
            v_mean, v_std = finite_difference_method(data, axis)

        # calculate the time step for the fast and slow scan directions
        # the fast scan direction can be calculated by the number of pixels per line and the scan rate
        # as the scn rate is trace + retrace, we need to multiply it by 2 to get the actual scan rate for one line
        # we also need to dived by the number of pixels per line to get the time per pixel
        # lastly we need to subtract 1 from the number of pixels per line to get the number of pixel steps
        dt_fast = 1 / (
            2 * scan_rate * (data.shape[axis] - 1) * (1 + scan_rounding)
        )  # s / pixel

        # the slow scan direction can be calculated directly by the inverse of the scan rate
        # we can do this as the tip moves once after each trace + retrace period
        dt_slow = 1 / scan_rate  # s

        # converting from nm / pixel to nm / s
        v_fast = v_mean / dt_fast  # nm / s
        v_fast_std = v_std / dt_fast  # nm / s

        v_slow = v_mean / dt_slow  # nm / s
        v_slow_std = v_std / dt_slow  # nm / s

        # calculate the estimated traveled distance in nm
        scan_distance = v_mean * data.shape[axis]
        scan_distance_std = v_std * data.shape[axis]

        # update the scan distances of the channels
        for channel in corrected_scan.channels.values():
            if channel.trace_direction == trace_direction:
                if channel_name.startswith("Xsensor"):
                    channel.scan_size = (scan_distance, channel.scan_size[1])
                    channel.scan_size_std = (
                        scan_distance_std,
                        channel.scan_size_std[1],
                    )
                elif channel_name.startswith("Ysensor"):
                    channel.scan_size = (channel.scan_size[0], scan_distance)
                    channel.scan_size_std = (
                        channel.scan_size_std[0],
                        scan_distance_std,
                    )

        corrected_scan.tip_velocity.update(
            {
                f"fast_{trace_direction}": v_fast.copy(),
                f"fast_std_{trace_direction}": v_fast_std.copy(),
                f"slow_{trace_direction}": v_slow.copy(),
                f"slow_std_{trace_direction}": v_slow_std.copy(),
            }
        )

    return corrected_scan


def extract_peaks(
    scan: Scan,
    channel_names: List[str] | None = None,
    pre_filter_size: int = 3,
    post_filter_size: int = 3,
    min_distance: int = 25,
    threshold_rel: float = 0.05,
    return_all_peaks: bool = False,
    inplace: bool = False,
    error_assumption: float = 0.5 / np.sqrt(3),
) -> Scan:
    if channel_names is None:
        channel_names = [
            channel_name
            for channel_name in scan.list_channels()
            if not (
                channel_name.startswith("Xsensor") or channel_name.startswith("Ysensor")
            )
        ]

    if not inplace:
        corrected_scan = scan.__copy__()
    else:
        corrected_scan = scan

    for channel_name in channel_names:
        data = corrected_scan.channels[channel_name].data
        scan_size = corrected_scan.channels[channel_name].scan_size

        # fft convolution is the fastest way to calculate the autocorrelation
        # correlation and convolution are the same, if the kernel is flipped for convolution
        # this is why whe use the reverse of the data for the kernel
        filtered_data = scipy.ndimage.gaussian_filter(data, pre_filter_size)
        autocorrelation = scipy.signal.fftconvolve(
            filtered_data, filtered_data[::-1, ::-1], mode="same"
        )

        # blur the autocorrelation to make the peaks more pronounced
        autocorrelation = scipy.ndimage.gaussian_filter(
            autocorrelation, post_filter_size
        )

        peaks = skimage.feature.peak_local_max(
            autocorrelation,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
            p_norm=2,
        )

        # sort the peaks by distance to the center
        center = np.array(autocorrelation.shape) / 2
        peak_distances = np.linalg.norm(peaks - center, axis=1)
        peaks = peaks[np.argsort(peak_distances)]

        # remove the peak with the lowest distance, as this is the center peak
        if not return_all_peaks:
            peaks = peaks[1:7]

        peaks = peaks[:, ::-1]

        sorted_indices = np.argsort(np.arctan2(peaks[:, 1], peaks[:, 0]))
        peaks = peaks[sorted_indices]

        # now interpolate the peak positions to nm
        to_nm_factor = np.array(scan_size) / np.array(data.shape)[::-1]
        peaks = peaks * to_nm_factor

        peak_std = error_assumption * to_nm_factor

        corrected_scan.channels[channel_name].peak_positions = peaks
        corrected_scan.channels[channel_name].peak_positions_std = peak_std

    return corrected_scan


def calculate_maximum_overlap_rectangle(
    scans: List[Scan],
    trace_direction: Literal["Trace", "Retrace"],
) -> Tuple[float, float, float, float]:
    def extract_rectangle_points(scans: List[Scan]) -> List[List[Tuple[float, float]]]:
        rect_points = []
        for scan in scans:
            x_sens = scan.channels[f"Xsensor: {trace_direction}"].data
            y_sens = scan.channels[f"Ysensor: {trace_direction}"].data
            x = np.min(x_sens)
            y = np.min(y_sens)
            w = np.max(x_sens) - x
            h = np.max(y_sens) - y
            rect_points.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        return rect_points

    def find_optimal_rectangle(
        rectangles: List[List[Tuple[float, float]]],
    ) -> Tuple[float, float, float, float]:
        rect_values = []

        max_mins = [(0, 1), (0, 0), (1, 0), (1, 1)]
        for i in range(4):
            # get the smallest x and y values
            x_min = min([rect[i][0] for rect in rectangles])
            x_max = max([rect[i][0] for rect in rectangles])
            y_min = min([rect[i][1] for rect in rectangles])
            y_max = max([rect[i][1] for rect in rectangles])

            values = [(x_min, y_min), (x_max, y_max)]
            x_value = values[max_mins[i][0]][0]
            y_value = values[max_mins[i][1]][1]

            if i % 2 == 1:
                rect_values.append(x_value)
            else:
                rect_values.append(y_value)

        x = rect_values[3]
        y = rect_values[0]
        w = rect_values[1] - x
        h = rect_values[2] - y

        return x, y, w, h

    if len(scans) == 1:
        scan = scans[0]
        x_sens = scan.channels[f"Xsensor: {trace_direction}"].data
        y_sens = scan.channels[f"Ysensor: {trace_direction}"].data
        x = np.min(x_sens)
        y = np.min(y_sens)
        w = np.max(x_sens) - x
        h = np.max(y_sens) - y
        return x, y, w, h

    rect_points = extract_rectangle_points(scans)
    x, y, w, h = find_optimal_rectangle(rect_points)

    return x, y, w, h


def interpolate_multiple_scans(
    scans: List[Scan],
    trace_direction: Literal["Trace", "Retrace"],
    channel_names: List[str],
    target_size: Tuple[int, int] = (512, 512),
    inplace: bool = False,
) -> List[Scan]:
    if not inplace:
        scans = [scan.__copy__() for scan in scans]

    x, y, w, h = calculate_maximum_overlap_rectangle(scans, trace_direction)
    new_X = np.linspace(
        x,
        x + w,
        target_size[0],
    )
    new_Y = np.linspace(
        y,
        y + h,
        target_size[1],
    )
    new_X, new_Y = np.meshgrid(new_X, new_Y)

    for scan in scans:
        x_sensor = scan.channels[f"Xsensor: {trace_direction}"].data
        y_sensor = scan.channels[f"Ysensor: {trace_direction}"].data
        X_flat = x_sensor.flatten()
        Y_flat = y_sensor.flatten()

        for channel_name in channel_names:
            if trace_direction not in channel_name:
                continue
            data = scan.channels[channel_name].data.flatten()
            interp = LinearNDInterpolator(
                list(zip(X_flat, Y_flat)), data, fill_value=data.mean()
            )
            interpolated_data = interp(new_X, new_Y)

            scan.channels[channel_name].data = interpolated_data
            scan.channels[channel_name].scan_size = (w, h)

        scan.channels[f"Xsensor: {trace_direction}"].data = new_X
        scan.channels[f"Ysensor: {trace_direction}"].data = new_Y
        scan.channels[f"Xsensor: {trace_direction}"].scan_size = (w, h)
        scan.channels[f"Ysensor: {trace_direction}"].scan_size = (w, h)

    return scans


def estimate_drift_from_cc(
    scans: list[Scan],
    channel_name: str,
    error_assumption: float = 0.5 / np.sqrt(3),
    peak_min_distance: int = 25,
    peak_threshold_rel: float = 0.05,
    peak_p_norm: int = 2,
):
    UP_scans = []
    DOWN_scans = []
    for scan in scans:
        scan_direction = scan.channels[channel_name].scan_direction
        if scan_direction == "Up":
            UP_scans.append(scan)
        elif scan_direction == "Down":
            DOWN_scans.append(scan)

    cc_drifts = []
    for scan_collection in [UP_scans, DOWN_scans]:
        for index, scan in enumerate(scan_collection):
            if index == len(scan_collection) - 1:
                break

            data_1 = scan_collection[index].channels[channel_name].data
            data_2 = scan_collection[index + 1].channels[channel_name].data
            scan_size = np.array(
                scan_collection[index].channels[channel_name].scan_size
            )
            scan_rate = scan_collection[index].scan_rate

            cross_corr = scipy.signal.fftconvolve(
                data_1, data_2[::-1, ::-1], mode="same"
            )
            center = np.array(cross_corr.shape) // 2
            peaks = skimage.feature.peak_local_max(
                cross_corr,
                min_distance=peak_min_distance,
                threshold_rel=peak_threshold_rel,
                p_norm=peak_p_norm,
            )

            peaks = np.array(peaks)
            center_peak = peaks[np.argmin(np.linalg.norm(peaks - center, axis=1))]

            cc_drift = center_peak - center
            cc_drift = cc_drift[::-1]

            cc_drift_x = u.ufloat(cc_drift[0], error_assumption)
            cc_drift_y = u.ufloat(cc_drift[1], error_assumption)
            cc_drift = np.array([cc_drift_x, cc_drift_y])

            to_nm_factor = scan_size / data_1.shape / (512 / scan_rate * 2)

            cc_drift = cc_drift * to_nm_factor
            smallest_possible_error = error_assumption * to_nm_factor

            cc_drifts.append(cc_drift)

    cc_dirft_mean = np.mean(cc_drifts, axis=0)

    cc_drift_x = cc_dirft_mean[0].n
    cc_drift_y = cc_dirft_mean[1].n
    cc_drift_x_error = max(smallest_possible_error[0], cc_dirft_mean[0].s)
    cc_drift_y_error = max(smallest_possible_error[1], cc_dirft_mean[1].s)

    return np.array([cc_drift_x, cc_drift_y]), np.array(
        [cc_drift_x_error, cc_drift_y_error]
    )
