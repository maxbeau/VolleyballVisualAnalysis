import numpy as np
from typing import List, Optional, Tuple


class KalmanRTS:
    """
    A generic Kalman Filter and Rauch-Tung-Striebel (RTS) smoother.
    Assumes a linear time-invariant system.
    """

    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray):
        """
        Initializes the Kalman filter and RTS smoother with system matrices.
        Args:
            F: State transition matrix.
            H: Observation matrix.
            Q: Process noise covariance.
            R: Measurement noise covariance.
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.n_state = F.shape[0]
        self.n_obs = H.shape[0]
        self.I = np.eye(self.n_state, dtype=float)

    def smooth(
        self,
        measurements: List[Optional[np.ndarray]],
        x0: np.ndarray,
        P0: np.ndarray,
        control_inputs: Optional[List[Optional[np.ndarray]]] = None,
    ) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """
        Performs forward filtering and backward smoothing.
        Args:
            measurements: List of measurements (np.ndarray) or None if no measurement.
            x0: Initial state estimate.
            P0: Initial error covariance.
            control_inputs: Optional list of control inputs for each time step.

        Returns:
            A tuple containing:
            - List of smoothed state estimates (or None).
            - List of smoothed error covariances (or None).
        """
        n_steps = len(measurements)
        
        # Forward pass (filtering)
        x_pred: List[Optional[np.ndarray]] = [None] * n_steps
        P_pred: List[Optional[np.ndarray]] = [None] * n_steps
        x_filt: List[Optional[np.ndarray]] = [None] * n_steps
        P_filt: List[Optional[np.ndarray]] = [None] * n_steps

        x_prev = x0
        P_prev = P0

        for j in range(n_steps):
            # Predict
            x_pr = self.F @ x_prev
            if control_inputs and control_inputs[j] is not None:
                x_pr += control_inputs[j]

            P_pr = self.F @ P_prev @ self.F.T + self.Q

            # Update
            z_j = measurements[j]
            if z_j is not None:
                y = z_j - (self.H @ x_pr)
                S = self.H @ P_pr @ self.H.T + self.R
                S += 1e-6 * np.eye(self.n_obs)  # for numerical stability
                try:
                    S_inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    S_inv = np.linalg.pinv(S)
                
                K = P_pr @ self.H.T @ S_inv
                x_upd = x_pr + K @ y
                P_upd = (self.I - K @ self.H) @ P_pr
            else:
                x_upd, P_upd = x_pr, P_pr

            x_pred[j], P_pred[j] = x_pr, P_pr
            x_filt[j], P_filt[j] = x_upd, P_upd
            x_prev, P_prev = x_upd, P_upd

        # Backward pass (smoothing)
        x_smooth: List[Optional[np.ndarray]] = [None] * n_steps
        P_smooth: List[Optional[np.ndarray]] = [None] * n_steps
        
        x_smooth[n_steps - 1] = x_filt[n_steps - 1]
        P_smooth[n_steps - 1] = P_filt[n_steps - 1]

        for j in range(n_steps - 2, -1, -1):
            xj_filt = x_filt[j]
            Pj_filt = P_filt[j]
            xj1_pred = x_pred[j + 1]
            Pj1_pred = P_pred[j + 1]
            xj1_smooth = x_smooth[j + 1]
            Pj1_smooth = P_smooth[j + 1]

            if xj_filt is None or Pj_filt is None or \
               xj1_pred is None or Pj1_pred is None or \
               xj1_smooth is None or Pj1_smooth is None:
                # If any part is missing, we can't continue the backward pass from here
                x_smooth[j] = xj_filt
                P_smooth[j] = Pj_filt
                continue

            try:
                Pj1_pred_inv = np.linalg.inv(Pj1_pred)
            except np.linalg.LinAlgError:
                Pj1_pred_inv = np.linalg.pinv(Pj1_pred)

            J = Pj_filt @ self.F.T @ Pj1_pred_inv
            x_smooth[j] = xj_filt + J @ (xj1_smooth - xj1_pred)
            P_smooth[j] = Pj_filt + J @ (Pj1_smooth - Pj1_pred) @ J.T

        return x_smooth, P_smooth