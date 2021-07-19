import torch

class KalmanFilter:
    def __init__(self, A, B, C, D, Q, R, N, init_mu, init_sigma):
        # Parameters that are different across time
        self.A = A
        self.C = C
        
        # Parameters that are shared across time
        self.B = B
        self.D = D
        self.Q = Q
        self.R = R
        self.N = N
        self.state_ndim = len(init_mu)
        
        # Predictive parameters: Allocate for N + 1 time steps and initialize at time 0
        self.mu_pred = torch.empty((N + 1, self.state_ndim))
        self.sigma_pred = torch.empty((N + 1, self.state_ndim, self.state_ndim))
        self.mu_pred[0] = init_mu
        self.sigma_pred[0] = init_sigma
        
        # Filtered parameters
        self.mu_filter = torch.empty((N + 1, self.state_ndim))
        self.sigma_filter = torch.empty((N + 1, self.state_ndim, self.state_ndim))
        
        # Smoothed parameters
        self.mu_smooth = torch.empty((N + 1, self.state_ndim))
        self.sigma_smooth = torch.empty((N + 1, self.state_ndim, self.state_ndim))
        
    #def _get_param(self, param, i):
    #    if len(params) == 1:
    #        return param[0]
    #    else:
    #        return param[i]
        
    def forward(self, y, y_gap=None, u=None):
        if y_gap is None:
            y_gap = 1 # fully observed
        if u is None:
            u = torch.zeros((self.N + 1, self.state_ndim)) # no input/control signal
            
        for i in range(self.N + 1):
            if i > 0:
                self.predict(u[i], self.A[i], self.B, self.Q, i)
            
            y_i = y[int(i / y_gap)] if y is not None and i % y_gap == 0 else None # is observed
            self.measure(y_i, u[i], self.C[i], self.D, self.R, i)
            
    def backward(self):
        # Initialize
        self.mu_smooth[self.N] = self.mu_filter[self.N]
        self.sigma_smooth[self.N] = self.sigma_filter[self.N]
        
        for i in range(self.N - 1, -1, -1):
            self.smooth(self.A[i+1], i)
        
    def predict(self, u, A, B, Q, i):
        self.mu_pred[i] = A @ self.mu_filter[i-1] + B @ u
        self.sigma_pred[i] = A @ self.sigma_filter[i-1] @ A.T + Q
        
    def measure(self, y, u, C, D, R, i):
        if y is not None:
            # Compute residual r
            y_hat = C @ self.mu_pred[i] + D @ u
            r = y - y_hat
            
            # Compute Kalman gain K
            S = C @ self.sigma_pred[i] @ C.T + R
            K = self.sigma_pred[i] @ C.T @ torch.inverse(S)
            
            # Update
            self.mu_filter[i] = self.mu_pred[i] + K @ r
            self.sigma_filter[i] = (torch.eye(self.state_ndim) - K @ C) @ self.sigma_pred[i]
        else:
            self.mu_filter[i] = self.mu_pred[i]
            self.sigma_filter[i] = self.sigma_pred[i]
        
    def smooth(self, A, i):
        # Compute backward Kalman gain J
        J = self.sigma_filter[i] @ A.T @ torch.inverse(self.sigma_pred[i+1])
        
        # Backward update
        self.mu_smooth[i] = self.mu_filter[i] + J @ (self.mu_smooth[i+1] - self.mu_pred[i+1])
        self.sigma_smooth[i] = self.sigma_filter[i] + J @ (self.sigma_smooth[i+1] - self.sigma_pred[i+1]) @ J.T
        