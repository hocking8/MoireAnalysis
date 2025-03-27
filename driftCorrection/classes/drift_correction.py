from typing import Literal

import numpy as np
import torch


from .scan import Scan


class DriftCorrectionModel(torch.nn.Module):
    def __init__(
        self,
        scans: list[Scan],
        channel_name: str,
        mode: Literal["fast_y", "fast_x"],
        lr: float = 1e-4,
    ) -> None:
        super(DriftCorrectionModel, self).__init__()

        self.mode = mode
        (
            self.peaks,
            self.peaks_std,
            self.v_fast,
            self.v_fast_std,
            self.v_slow,
            self.v_slow_std,
        ) = self.precompute_peaks_and_velocities(scans, channel_name)

        self.drift = torch.nn.Parameter(
            torch.tensor(
                [0.0, 0.0],
                requires_grad=True,
            ),
        )
        self.register_parameter("drift", self.drift)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.loss_function = torch.nn.MSELoss()
        self.losses: np.ndarray | None = None
        self.drifts: np.ndarray | None = None
        self.final_drift: np.ndarray | None = None
        self.final_loss: float | None = None

    def _optimize(self) -> None:
        self.optimizer.zero_grad()
        output = self.forward()
        loss = self.loss_function(output, torch.zeros_like(output))
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        self.drifts.append(self.drift.data.numpy().copy())

    def fit(
        self,
        epochs: int = 100,
        verbose: bool = False,
    ) -> None:
        self.drifts = []
        self.losses = []

        for epoch in range(epochs):
            self._optimize()
            if verbose:
                print(
                    f"Epoch {epoch + 1:5}/{epochs:5}, Loss: {self.losses[-1]:.4f}",
                    end="\r",
                )
        self.drifts = np.array(self.drifts)
        self.losses = np.array(self.losses)

        self.final_drift = self.drift.data.numpy().copy()
        self.final_loss = self.losses[-1]

    @staticmethod
    def precompute_peaks_and_velocities(
        scans: list[Scan],
        channel_name: str,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        all_peaks = []
        all_peaks_std = []
        all_v_fast = []
        all_v_fast_std = []
        all_v_slow = []
        all_v_slow_std = []

        trace_direction = channel_name.split(":")[1].strip()
        for scan in scans:
            x_sensor = scan.channels[f"Xsensor: {trace_direction}"].data
            y_sensor = scan.channels[f"Ysensor: {trace_direction}"].data

            center = np.array([x_sensor.mean(), y_sensor.mean()])
            peaks = np.array(scan.channels[channel_name].peak_positions)
            peaks_std = np.array(scan.channels[channel_name].peak_positions_std)

            # we need to center the peaks, so we get the right values for the drift matrix
            peaks = peaks - center
            peaks = torch.tensor(peaks, dtype=torch.float32)
            peaks_std = torch.tensor(peaks_std, dtype=torch.float32)

            v_fast = torch.tensor(
                scan.tip_velocity[f"fast_{trace_direction}"],
                dtype=torch.float32,
            )
            v_fast_std = torch.tensor(
                scan.tip_velocity[f"fast_std_{trace_direction}"],
                dtype=torch.float32,
            )

            v_slow = torch.tensor(
                scan.tip_velocity[f"slow_{trace_direction}"],
                dtype=torch.float32,
            )
            v_slow_std = torch.tensor(
                scan.tip_velocity[f"slow_std_{trace_direction}"],
                dtype=torch.float32,
            )

            all_peaks.append(peaks)
            all_peaks_std.append(peaks_std)
            all_v_fast.append(v_fast)
            all_v_fast_std.append(v_fast_std)
            all_v_slow.append(v_slow)
            all_v_slow_std.append(v_slow_std)

        return (
            all_peaks,
            all_peaks_std,
            all_v_fast,
            all_v_fast_std,
            all_v_slow,
            all_v_slow_std,
        )

    @staticmethod
    def drift_matrix(
        V_fast: float,
        V_slow: float,
        D_x: float,
        D_y: float,
        mode: Literal["fast_y", "fast_x"],
    ) -> np.ndarray:
        if mode == "fast_x":
            D_s = D_y
            D_f = D_x

            alpha_s = D_s / (V_slow - D_s)
            alpha_f = D_f / (V_fast - D_f)

            A_11 = 1 + alpha_f
            A_12 = D_f / (V_slow - D_s) * (1 + alpha_f)
            A_21 = D_s / (V_fast - D_f)
            A_22 = 1 + alpha_s * (1 + alpha_f)

        elif mode == "fast_y":
            D_s = D_x
            D_f = D_y

            alpha_s = D_s / (V_slow - D_s)
            alpha_f = D_f / (V_fast - D_f)

            A_11 = 1 + alpha_s * (1 + alpha_f)
            A_12 = D_s / (V_fast - D_f)
            A_21 = D_f / (V_slow - D_s) * (1 + alpha_f)
            A_22 = 1 + alpha_f

        else:
            raise ValueError("Invalid mode")

        return np.array([[A_11, A_12], [A_21, A_22]])

    def drift_matrix_torch(
        self,
        V_fast: torch.Tensor,
        V_slow: torch.Tensor,
        mode: Literal["fast_y", "fast_x"],
    ) -> torch.Tensor:
        if mode == "fast_x":
            D_s = self.drift[1]
            D_f = self.drift[0]

            alpha_s = D_s / (V_slow - D_s)
            alpha_f = D_f / (V_fast - D_f)

            A_11 = 1 + alpha_f
            A_12 = D_f / (V_slow - D_s) * (1 + alpha_f)
            A_21 = D_s / (V_fast - D_f)
            A_22 = 1 + alpha_s * (1 + alpha_f)

        elif mode == "fast_y":
            D_s = self.drift[0]
            D_f = self.drift[1]

            alpha_s = D_s / (V_slow - D_s)
            alpha_f = D_f / (V_fast - D_f)

            A_11 = 1 + alpha_s * (1 + alpha_f)
            A_12 = D_s / (V_fast - D_f)
            A_21 = D_f / (V_slow - D_s) * (1 + alpha_f)
            A_22 = 1 + alpha_f

        else:
            raise ValueError("Invalid mode")

        # keep the computation graph intact
        # we need to stack the tensors to get a 2x2 matrix
        # we cant just create a 2x2 tensor with the values, this would wipe the history of the values
        return torch.stack([torch.stack([A_11, A_12]), torch.stack([A_21, A_22])])

    def forward(self) -> torch.Tensor:
        output = []

        # DOWN; UP pairs
        # i.e. All scans here are
        # (DOWN, UP), (DOWN, UP), ...
        for index in range(0, len(self.peaks) - 1, 2):
            peaks_a = self.peaks[index]
            v_fast_a = self.v_fast[index]
            v_slow_a = self.v_slow[index]

            peaks_b = self.peaks[index + 1]
            v_fast_b = self.v_fast[index + 1]
            v_slow_b = self.v_slow[index + 1]

            # here we are using the forward drift matrix
            A_inv_a = torch.inverse(
                self.drift_matrix_torch(
                    v_fast_a,
                    v_slow_a,
                    self.mode,
                )
            )
            A_inv_b = torch.inverse(
                self.drift_matrix_torch(
                    v_fast_b,
                    -v_slow_b,  # watch the negative sign here!
                    self.mode,
                )
            )

            # as the peaks are ordered in the same way we can compare one by one
            for peak_a, peak_b in zip(peaks_a, peaks_b):
                transformed_a = A_inv_a @ peak_a
                transformed_b = A_inv_b @ peak_b
                diff = torch.norm(transformed_a - transformed_b, p=2)
                output.append(diff)

        # UP; DOWN pairs
        # i.e. All scans here are
        # (UP, DOWN), (UP, DOWN), ...
        for index in range(1, len(self.peaks) - 1, 2):
            peaks_a = self.peaks[index]
            v_fast_a = self.v_fast[index]
            v_slow_a = self.v_slow[index]

            peaks_b = self.peaks[index + 1]
            v_fast_b = self.v_fast[index + 1]
            v_slow_b = self.v_slow[index + 1]

            # here we are using the backward drift matrix
            A_inv_a = torch.inverse(
                self.drift_matrix_torch(
                    v_fast_a,
                    -v_slow_a,  # watch the negative sign here!
                    self.mode,
                )
            )
            A_inv_b = torch.inverse(
                self.drift_matrix_torch(
                    v_fast_b,
                    v_slow_b,
                    self.mode,
                )
            )

            for peak_a, peak_b in zip(peaks_a, peaks_b):
                transformed_a = A_inv_a @ peak_a
                transformed_b = A_inv_b @ peak_b
                diff = torch.norm(transformed_a - transformed_b, p=2)
                output.append(diff)

        return torch.stack(output)


class DriftCorrectionModelMC:
    def __init__(
        self,
        scans: list[Scan],
        channel_name: str,
        mode: Literal["fast_y", "fast_x"],
        lr: float = 5e-4,
        num_samples: int = 1000,
    ) -> None:
        self.mode = mode
        self.lr = lr
        (
            self.peaks,
            self.peaks_std,
            self.v_fast,
            self.v_fast_std,
            self.v_slow,
            self.v_slow_std,
        ) = self.precompute_peaks_and_velocities(scans, channel_name)

        self.loss_function = torch.nn.MSELoss()
        self.mc_drifts: np.ndarray | None = None
        self.losses: np.ndarray | None = None
        self.drifts: np.ndarray | None = None
        self.final_drift: np.ndarray | None = None
        self.final_loss: float | None = None
        self.num_samples = num_samples
        self.sampled_peaks = torch.zeros((self.num_samples, *self.peaks.shape))
        self.sampled_v_slow = torch.zeros((self.num_samples, *self.v_slow.shape))
        self.sampled_v_fast = torch.zeros((self.num_samples, *self.v_fast.shape))

        self._sample_variables()

    def _sample_variables(self) -> None:
        for i, (peak, peak_std) in enumerate(zip(self.peaks, self.peaks_std)):
            for j, p in enumerate(peak):
                sample = np.random.multivariate_normal(
                    mean=p.numpy(),
                    cov=np.diag(peak_std.numpy()) ** 2,
                    size=self.num_samples,
                )
                sample = torch.tensor(sample, dtype=torch.float32)
                for k, s in enumerate(sample):
                    self.sampled_peaks[k, i, j] = s

        for i, (v, v_std) in enumerate(zip(self.v_fast, self.v_fast_std)):
            sample = np.random.normal(loc=v, scale=v_std, size=self.num_samples)
            sample = torch.tensor(sample, dtype=torch.float32)
            for k, s in enumerate(sample):
                self.sampled_v_fast[k, i] = s

        for i, (v, v_std) in enumerate(zip(self.v_slow, self.v_slow_std)):
            sample = np.random.normal(loc=v, scale=v_std, size=self.num_samples)
            sample = torch.tensor(sample, dtype=torch.float32)
            for k, s in enumerate(sample):
                self.sampled_v_slow[k, i] = s

    def _optimize(
        self,
        epochs: int,
        peaks: torch.Tensor,
        v_fast: torch.Tensor,
        v_slow: torch.Tensor,
        lr: float,
        stop_tol: float | None = None,
    ) -> None:
        drift = torch.nn.Parameter(
            torch.tensor(
                [0.0, 0.0],
                requires_grad=True,
            ),
        )

        optimizer = torch.optim.SGD([drift], lr=lr)
        loss_function = torch.nn.MSELoss()

        losses = []
        drifts = []

        for _ in range(epochs):
            optimizer.zero_grad()
            output = self.forward(
                drift,
                peaks,
                v_fast,
                v_slow,
            )

            loss = loss_function(output, torch.zeros_like(output))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            drifts.append(drift.data.numpy().copy())

            if stop_tol is not None and len(losses) > 1:
                if np.abs(losses[-1] - losses[-2]) < stop_tol:
                    break

        drift = drift.data.numpy().copy()

        return drift, np.array(losses), np.array(drifts)

    def fit(
        self,
        epochs: int = 100,
        verbose: bool = False,
        stop_tol: float | None = None,
    ) -> None:
        self.mc_drifts = []
        self.drifts = []
        self.losses = []

        for sample_index in range(self.num_samples):
            opt_drift, s_losses, s_drifts = self._optimize(
                epochs=epochs,
                peaks=self.sampled_peaks[sample_index],
                v_fast=self.sampled_v_fast[sample_index],
                v_slow=self.sampled_v_slow[sample_index],
                lr=self.lr,
                stop_tol=stop_tol,
            )
            if verbose:
                print(
                    f"Sample {sample_index + 1:5}/{self.num_samples:5}, Loss: {s_losses[-1]:.4f}",
                    end="\r",
                )
            self.losses.append(s_losses)
            self.drifts.append(s_drifts)
            self.mc_drifts.append(opt_drift)

        self.mc_drifts = np.array(self.mc_drifts)
        self.drifts = np.array(self.drifts)
        self.losses = np.array(self.losses)

        opt_drift, s_losses, s_drifts = self._optimize(
            epochs=epochs,
            peaks=self.peaks,
            v_fast=self.v_fast,
            v_slow=self.v_slow,
            lr=self.lr,
            stop_tol=stop_tol,
        )

        self.mean_drift = np.mean(self.mc_drifts, axis=0)
        self.std_drift = np.std(self.mc_drifts, axis=0)
        self.final_drift = opt_drift

    @staticmethod
    def precompute_peaks_and_velocities(
        scans: list[Scan],
        channel_name: str,
        as_numpy: bool = False,
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        | tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]
    ):
        all_peaks = []
        all_peaks_std = []
        all_v_fast = []
        all_v_fast_std = []
        all_v_slow = []
        all_v_slow_std = []

        trace_direction = channel_name.split(":")[1].strip()
        for scan in scans:
            x_sensor = scan.channels[f"Xsensor: {trace_direction}"].data
            y_sensor = scan.channels[f"Ysensor: {trace_direction}"].data

            center = np.array(
                [
                    x_sensor.mean() - x_sensor.min(),
                    y_sensor.mean() - y_sensor.min(),
                ]
            )
            peaks = np.array(scan.channels[channel_name].peak_positions)
            peaks_std = np.array(scan.channels[channel_name].peak_positions_std)

            # we need to center the peaks, so we get the right values for the drift matrix
            peaks = peaks - center
            peaks = torch.tensor(peaks, dtype=torch.float32)
            peaks_std = torch.tensor(peaks_std, dtype=torch.float32)

            v_fast = torch.tensor(
                scan.tip_velocity[f"fast_{trace_direction}"],
                dtype=torch.float32,
            )
            v_fast_std = torch.tensor(
                scan.tip_velocity[f"fast_std_{trace_direction}"],
                dtype=torch.float32,
            )

            v_slow = torch.tensor(
                scan.tip_velocity[f"slow_{trace_direction}"],
                dtype=torch.float32,
            )
            v_slow_std = torch.tensor(
                scan.tip_velocity[f"slow_std_{trace_direction}"],
                dtype=torch.float32,
            )

            all_peaks.append(peaks)
            all_peaks_std.append(peaks_std)
            all_v_fast.append(v_fast)
            all_v_fast_std.append(v_fast_std)
            all_v_slow.append(v_slow)
            all_v_slow_std.append(v_slow_std)

        all_peaks = torch.stack(all_peaks)
        all_peaks_std = torch.stack(all_peaks_std)
        all_v_fast = torch.stack(all_v_fast)
        all_v_fast_std = torch.stack(all_v_fast_std)
        all_v_slow = torch.stack(all_v_slow)
        all_v_slow_std = torch.stack(all_v_slow_std)

        if as_numpy:
            all_peaks = all_peaks.numpy()
            all_peaks_std = all_peaks_std.numpy()
            all_v_fast = all_v_fast.numpy()
            all_v_fast_std = all_v_fast_std.numpy()
            all_v_slow = all_v_slow.numpy()
            all_v_slow_std = all_v_slow_std.numpy()

        return (
            all_peaks,
            all_peaks_std,
            all_v_fast,
            all_v_fast_std,
            all_v_slow,
            all_v_slow_std,
        )

    @staticmethod
    def drift_matrix(
        V_fast: float,
        V_slow: float,
        D_x: float,
        D_y: float,
        mode: Literal["fast_y", "fast_x"],
    ) -> np.ndarray:
        if mode == "fast_x":
            D_s = D_y
            D_f = D_x

            alpha_s = D_s / (V_slow - D_s)
            alpha_f = D_f / (V_fast - D_f)

            A_11 = 1 + alpha_f
            A_12 = D_f / (V_slow - D_s) * (1 + alpha_f)
            A_21 = D_s / (V_fast - D_f)
            A_22 = 1 + alpha_s * (1 + alpha_f)

        elif mode == "fast_y":
            D_s = D_x
            D_f = D_y

            alpha_s = D_s / (V_slow - D_s)
            alpha_f = D_f / (V_fast - D_f)

            A_11 = 1 + alpha_s * (1 + alpha_f)
            A_12 = D_s / (V_fast - D_f)
            A_21 = D_f / (V_slow - D_s) * (1 + alpha_f)
            A_22 = 1 + alpha_f

        else:
            raise ValueError("Invalid mode")

        return np.array([[A_11, A_12], [A_21, A_22]])

    @staticmethod
    def drift_matrix_torch(
        drift: torch.Tensor,
        V_fast: torch.Tensor,
        V_slow: torch.Tensor,
        mode: Literal["fast_y", "fast_x"],
    ) -> torch.Tensor:
        if mode == "fast_x":
            D_s = drift[1]
            D_f = drift[0]

            alpha_s = D_s / (V_slow - D_s)
            alpha_f = D_f / (V_fast - D_f)

            A_11 = 1 + alpha_f
            A_12 = D_f / (V_slow - D_s) * (1 + alpha_f)
            A_21 = D_s / (V_fast - D_f)
            A_22 = 1 + alpha_s * (1 + alpha_f)

        elif mode == "fast_y":
            D_s = drift[0]
            D_f = drift[1]

            alpha_s = D_s / (V_slow - D_s)
            alpha_f = D_f / (V_fast - D_f)

            A_11 = 1 + alpha_s * (1 + alpha_f)
            A_12 = D_s / (V_fast - D_f)
            A_21 = D_f / (V_slow - D_s) * (1 + alpha_f)
            A_22 = 1 + alpha_f

        else:
            raise ValueError("Invalid mode")

        # keep the computation graph intact
        # we need to stack the tensors to get a 2x2 matrix
        # we cant just create a 2x2 tensor with the values, this would wipe the history of the values
        return torch.stack([torch.stack([A_11, A_12]), torch.stack([A_21, A_22])])

    def forward(
        self,
        drift: torch.Tensor,
        peaks: list[torch.Tensor],
        v_fast: list[torch.Tensor],
        v_slow: list[torch.Tensor],
    ) -> torch.Tensor:
        output = []

        # DOWN; UP pairs
        # i.e. All scans here are
        # (DOWN, UP), (DOWN, UP), ...
        for index in range(0, len(peaks) - 1, 2):
            peaks_a = peaks[index]
            v_fast_a = v_fast[index]
            v_slow_a = v_slow[index]

            peaks_b = peaks[index + 1]
            v_fast_b = v_fast[index + 1]
            v_slow_b = v_slow[index + 1]

            # here we are using the forward drift matrix
            A_inv_a = torch.inverse(
                self.drift_matrix_torch(
                    drift,
                    v_fast_a,
                    v_slow_a,
                    self.mode,
                )
            )
            A_inv_b = torch.inverse(
                self.drift_matrix_torch(
                    drift,
                    v_fast_b,
                    -v_slow_b,  # watch the negative sign here!
                    self.mode,
                )
            )

            # as the peaks are ordered in the same way we can compare one by one
            for peak_a, peak_b in zip(peaks_a, peaks_b):
                transformed_a = A_inv_a @ peak_a
                transformed_b = A_inv_b @ peak_b
                diff = torch.norm(transformed_a - transformed_b, p=2)
                output.append(diff)

        # UP; DOWN pairs
        # i.e. All scans here are
        # (UP, DOWN), (UP, DOWN), ...
        for index in range(1, len(self.peaks) - 1, 2):
            peaks_a = peaks[index]
            v_fast_a = v_fast[index]
            v_slow_a = v_slow[index]

            peaks_b = peaks[index + 1]
            v_fast_b = v_fast[index + 1]
            v_slow_b = v_slow[index + 1]

            # here we are using the backward drift matrix
            A_inv_a = torch.inverse(
                self.drift_matrix_torch(
                    drift,
                    v_fast_a,
                    -v_slow_a,  # watch the negative sign here!
                    self.mode,
                )
            )
            A_inv_b = torch.inverse(
                self.drift_matrix_torch(
                    drift,
                    v_fast_b,
                    v_slow_b,
                    self.mode,
                )
            )

            for peak_a, peak_b in zip(peaks_a, peaks_b):
                transformed_a = A_inv_a @ peak_a
                transformed_b = A_inv_b @ peak_b
                diff = torch.norm(transformed_a - transformed_b, p=2)
                output.append(diff)

        return torch.stack(output)


class DriftCorrectionModelMC:
    def __init__(
        self,
        scans: list[Scan],
        channel_name: str,
        mode: Literal["fast_y", "fast_x"],
        lr: float = 5e-4,
        num_samples: int = 1000,
    ) -> None:
        self.mode = mode
        self.lr = lr
        (
            self.peaks,
            self.peaks_std,
            self.v_fast,
            self.v_fast_std,
            self.v_slow,
            self.v_slow_std,
        ) = self.precompute_peaks_and_velocities(scans, channel_name)

        self.loss_function = torch.nn.MSELoss()
        self.mc_drifts: np.ndarray | None = None
        self.losses: np.ndarray | None = None
        self.drifts: np.ndarray | None = None
        self.final_drift: np.ndarray | None = None
        self.final_loss: float | None = None
        self.num_samples = num_samples
        self.sampled_peaks = torch.zeros((self.num_samples, *self.peaks.shape))
        self.sampled_v_slow = torch.zeros((self.num_samples, *self.v_slow.shape))
        self.sampled_v_fast = torch.zeros((self.num_samples, *self.v_fast.shape))

        self._sample_variables()

    def _sample_variables(self) -> None:
        for i, (peak, peak_std) in enumerate(zip(self.peaks, self.peaks_std)):
            for j, p in enumerate(peak):
                sample = np.random.multivariate_normal(
                    mean=p.numpy(),
                    cov=np.diag(peak_std.numpy()) ** 2,
                    size=self.num_samples,
                )
                sample = torch.tensor(sample, dtype=torch.float32)
                for k, s in enumerate(sample):
                    self.sampled_peaks[k, i, j] = s

        for i, (v, v_std) in enumerate(zip(self.v_fast, self.v_fast_std)):
            sample = np.random.normal(loc=v, scale=v_std, size=self.num_samples)
            sample = torch.tensor(sample, dtype=torch.float32)
            for k, s in enumerate(sample):
                self.sampled_v_fast[k, i] = s

        for i, (v, v_std) in enumerate(zip(self.v_slow, self.v_slow_std)):
            sample = np.random.normal(loc=v, scale=v_std, size=self.num_samples)
            sample = torch.tensor(sample, dtype=torch.float32)
            for k, s in enumerate(sample):
                self.sampled_v_slow[k, i] = s

    def _optimize(
        self,
        epochs: int,
        peaks: torch.Tensor,
        v_fast: torch.Tensor,
        v_slow: torch.Tensor,
        lr: float,
        stop_tol: float | None = None,
    ) -> None:
        drift = torch.nn.Parameter(
            torch.tensor(
                [0.0, 0.0],
                requires_grad=True,
            ),
        )

        optimizer = torch.optim.SGD([drift], lr=lr)
        loss_function = torch.nn.MSELoss()

        losses = []
        drifts = []

        for _ in range(epochs):
            optimizer.zero_grad()
            output = self.forward(
                drift,
                peaks,
                v_fast,
                v_slow,
            )

            loss = loss_function(output, torch.zeros_like(output))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            drifts.append(drift.data.numpy().copy())

            if stop_tol is not None and len(losses) > 1:
                if np.abs(losses[-1] - losses[-2]) < stop_tol:
                    break

        drift = drift.data.numpy().copy()

        return drift, np.array(losses), np.array(drifts)

    def fit(
        self,
        epochs: int = 100,
        verbose: bool = False,
        stop_tol: float | None = None,
    ) -> None:
        self.mc_drifts = []
        self.drifts = []
        self.losses = []

        for sample_index in range(self.num_samples):
            opt_drift, s_losses, s_drifts = self._optimize(
                epochs=epochs,
                peaks=self.sampled_peaks[sample_index],
                v_fast=self.sampled_v_fast[sample_index],
                v_slow=self.sampled_v_slow[sample_index],
                lr=self.lr,
                stop_tol=stop_tol,
            )
            if verbose:
                print(
                    f"Sample {sample_index + 1:5}/{self.num_samples:5}, Loss: {s_losses[-1]:.4f}",
                    end="\r",
                )
            self.losses.append(s_losses)
            self.drifts.append(s_drifts)
            self.mc_drifts.append(opt_drift)

        self.mc_drifts = np.array(self.mc_drifts)
        self.drifts = np.array(self.drifts)
        self.losses = np.array(self.losses)

        opt_drift, s_losses, s_drifts = self._optimize(
            epochs=epochs,
            peaks=self.peaks,
            v_fast=self.v_fast,
            v_slow=self.v_slow,
            lr=self.lr,
            stop_tol=stop_tol,
        )

        self.mean_drift = np.mean(self.mc_drifts, axis=0)
        self.std_drift = np.std(self.mc_drifts, axis=0)
        self.final_drift = opt_drift

    @staticmethod
    def precompute_peaks_and_velocities(
        scans: list[Scan],
        channel_name: str,
        as_numpy: bool = False,
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        | tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]
    ):
        all_peaks = []
        all_peaks_std = []
        all_v_fast = []
        all_v_fast_std = []
        all_v_slow = []
        all_v_slow_std = []

        trace_direction = channel_name.split(":")[1].strip()
        for scan in scans:
            x_sensor = scan.channels[f"Xsensor: {trace_direction}"].data
            y_sensor = scan.channels[f"Ysensor: {trace_direction}"].data

            center = np.array(
                [
                    x_sensor.mean() - x_sensor.min(),
                    y_sensor.mean() - y_sensor.min(),
                ]
            )
            peaks = np.array(scan.channels[channel_name].peak_positions)
            peaks_std = np.array(scan.channels[channel_name].peak_positions_std)

            # we need to center the peaks, so we get the right values for the drift matrix
            peaks = peaks - center
            peaks = torch.tensor(peaks, dtype=torch.float32)
            peaks_std = torch.tensor(peaks_std, dtype=torch.float32)

            v_fast = torch.tensor(
                scan.tip_velocity[f"fast_{trace_direction}"],
                dtype=torch.float32,
            )
            v_fast_std = torch.tensor(
                scan.tip_velocity[f"fast_std_{trace_direction}"],
                dtype=torch.float32,
            )

            v_slow = torch.tensor(
                scan.tip_velocity[f"slow_{trace_direction}"],
                dtype=torch.float32,
            )
            v_slow_std = torch.tensor(
                scan.tip_velocity[f"slow_std_{trace_direction}"],
                dtype=torch.float32,
            )

            all_peaks.append(peaks)
            all_peaks_std.append(peaks_std)
            all_v_fast.append(v_fast)
            all_v_fast_std.append(v_fast_std)
            all_v_slow.append(v_slow)
            all_v_slow_std.append(v_slow_std)

        all_peaks = torch.stack(all_peaks)
        all_peaks_std = torch.stack(all_peaks_std)
        all_v_fast = torch.stack(all_v_fast)
        all_v_fast_std = torch.stack(all_v_fast_std)
        all_v_slow = torch.stack(all_v_slow)
        all_v_slow_std = torch.stack(all_v_slow_std)

        if as_numpy:
            all_peaks = all_peaks.numpy()
            all_peaks_std = all_peaks_std.numpy()
            all_v_fast = all_v_fast.numpy()
            all_v_fast_std = all_v_fast_std.numpy()
            all_v_slow = all_v_slow.numpy()
            all_v_slow_std = all_v_slow_std.numpy()

        return (
            all_peaks,
            all_peaks_std,
            all_v_fast,
            all_v_fast_std,
            all_v_slow,
            all_v_slow_std,
        )

    @staticmethod
    def drift_matrix(
        V_fast: float,
        V_slow: float,
        D_x: float,
        D_y: float,
        mode: Literal["fast_y", "fast_x"],
    ) -> np.ndarray:
        if mode == "fast_x":
            D_s = D_y
            D_f = D_x

            alpha_s = D_s / (V_slow - D_s)
            alpha_f = D_f / (V_fast - D_f)

            A_11 = 1 + alpha_f
            A_12 = D_f / (V_slow - D_s) * (1 + alpha_f)
            A_21 = 0.0
            A_22 = 1 + alpha_s

        elif mode == "fast_y":
            D_s = D_x
            D_f = D_y

            alpha_s = D_s / (V_slow - D_s)
            alpha_f = D_f / (V_fast - D_f)

            A_11 = 1 + alpha_s
            A_12 = 0.0
            A_21 = D_f / (V_slow - D_s) * (1 + alpha_f)
            A_22 = 1 + alpha_f

        else:
            raise ValueError("Invalid mode")

        return np.array([[A_11, A_12], [A_21, A_22]])

    @staticmethod
    def drift_matrix_torch(
        drift: torch.Tensor,
        V_fast: torch.Tensor,
        V_slow: torch.Tensor,
        mode: Literal["fast_y", "fast_x"],
    ) -> torch.Tensor:
        if mode == "fast_x":
            D_s = drift[1]
            D_f = drift[0]

            alpha_s = D_s / (V_slow - D_s)
            alpha_f = D_f / (V_fast - D_f)

            A_11 = 1 + alpha_f
            A_12 = D_f / (V_slow - D_s) * (1 + alpha_f)
            A_21 = torch.tensor(0.0)
            A_22 = 1 + alpha_s

        elif mode == "fast_y":
            D_s = drift[0]
            D_f = drift[1]

            alpha_s = D_s / (V_slow - D_s)
            alpha_f = D_f / (V_fast - D_f)

            A_11 = 1 + alpha_s
            A_12 = torch.tensor(0.0)
            A_21 = D_f / (V_slow - D_s) * (1 + alpha_f)
            A_22 = 1 + alpha_f

        else:
            raise ValueError("Invalid mode")

        # keep the computation graph intact
        # we need to stack the tensors to get a 2x2 matrix
        # we cant just create a 2x2 tensor with the values, this would wipe the history of the values
        return torch.stack([torch.stack([A_11, A_12]), torch.stack([A_21, A_22])])

    def forward(
        self,
        drift: torch.Tensor,
        peaks: list[torch.Tensor],
        v_fast: list[torch.Tensor],
        v_slow: list[torch.Tensor],
    ) -> torch.Tensor:
        output = []

        # DOWN; UP pairs
        # i.e. All scans here are
        # (DOWN, UP), (DOWN, UP), ...
        for index in range(0, len(peaks) - 1, 2):
            peaks_a = peaks[index]
            v_fast_a = v_fast[index]
            v_slow_a = v_slow[index]

            peaks_b = peaks[index + 1]
            v_fast_b = v_fast[index + 1]
            v_slow_b = v_slow[index + 1]

            # here we are using the forward drift matrix
            A_inv_a = torch.inverse(
                self.drift_matrix_torch(
                    drift,
                    v_fast_a,
                    v_slow_a,
                    self.mode,
                )
            )
            A_inv_b = torch.inverse(
                self.drift_matrix_torch(
                    drift,
                    v_fast_b,
                    -v_slow_b,  # watch the negative sign here!
                    self.mode,
                )
            )

            # as the peaks are ordered in the same way we can compare one by one
            for peak_a, peak_b in zip(peaks_a, peaks_b):
                transformed_a = A_inv_a @ peak_a
                transformed_b = A_inv_b @ peak_b
                diff = torch.norm(transformed_a - transformed_b, p=2)
                output.append(diff)

        # UP; DOWN pairs
        # i.e. All scans here are
        # (UP, DOWN), (UP, DOWN), ...
        for index in range(1, len(self.peaks) - 1, 2):
            peaks_a = peaks[index]
            v_fast_a = v_fast[index]
            v_slow_a = v_slow[index]

            peaks_b = peaks[index + 1]
            v_fast_b = v_fast[index + 1]
            v_slow_b = v_slow[index + 1]

            # here we are using the backward drift matrix
            A_inv_a = torch.inverse(
                self.drift_matrix_torch(
                    drift,
                    v_fast_a,
                    -v_slow_a,  # watch the negative sign here!
                    self.mode,
                )
            )
            A_inv_b = torch.inverse(
                self.drift_matrix_torch(
                    drift,
                    v_fast_b,
                    v_slow_b,
                    self.mode,
                )
            )

            for peak_a, peak_b in zip(peaks_a, peaks_b):
                transformed_a = A_inv_a @ peak_a
                transformed_b = A_inv_b @ peak_b
                diff = torch.norm(transformed_a - transformed_b, p=2)
                output.append(diff)

        return torch.stack(output)
