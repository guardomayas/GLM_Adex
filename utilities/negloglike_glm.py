import numpy as np

# def negloglike_glm_basis(prs, NL, xconvki, yconvhi, y, dt, refreshRate, L2pen=0):
#     """
#     Calculate negative log likelihood, gradient, and Hessian for a Poisson GLM.
    
#     Parameters
#     ----------
#     prs : np.array
#         Vector of parameters, coefficients for basis functions [kprs, hprs, dc]
#     NL : function
#         Nonlinearity (e.g., np.exp)
#     xconvki : np.array
#         Stimulus convolved with each filter basis vector
#     yconvhi : np.array
#         Response convolved with each post-spike filter basis vector
#     y : np.array
#         Response vector (spike train of 0s and 1s)
#     dt : float
#         Time step of y (in frames/stimulus frame)
#     refreshRate : float
#         Refresh rate of stimulus (frames/sec)
#     L2pen : float, optional
#         L2 penalty on parameters (default is 0)
    
#     Returns
#     -------
#     negloglike : float
#         Negative log-likelihood of the GLM
#     dL : np.array
#         Gradient of the log-likelihood with respect to parameters
#     H : np.array
#         Hessian of the log-likelihood with respect to parameters
#     """
#     nkbasis = xconvki.shape[1]  # Number of basis functions for k
#     kprs = prs[:nkbasis]  # k basis functions
#     hprs = prs[nkbasis:-1]  # h basis functions
#     dc = prs[-1]  # DC term

#     # Calculate xconvk_dc and yconvh
#     xconvk_dc = np.dot(xconvki, kprs) + dc  # Shape (T,)
#     yconvh = np.dot(yconvhi, hprs)  # Shape (T,)
#     g = xconvk_dc + yconvh  # Shape (T,)

#     # Apply the nonlinearity
#     lambda_pred = NL(g)  # Shape (T,)

#     # Calculate the negative log-likelihood
#     negloglike = -np.dot(y, g) + dt * np.sum(lambda_pred) / refreshRate + L2pen * np.dot(prs, prs)
#     # negloglike = -np.dot(y, g) + dt * np.sum(lambda_pred) / refreshRate + L2pen * np.dot(prs[:-1], prs[:-1]) #exclude dc
#     print(f"Negative log-likelihood: {negloglike}")
#     ## Gradient Calculation
#     dL = np.zeros_like(prs)
#     prsMat = np.hstack([xconvki, yconvhi, np.ones((xconvki.shape[0], 1))])  # Shape (T, nkbasis + nhbasis + 1)

#     for pr in range(len(prs)):
#         dL[pr] = -np.sum(prsMat[y.astype(bool), pr]) + dt / refreshRate * np.sum(prsMat[:, pr] * lambda_pred) + L2pen * 2 * prs[pr]

#     ## Hessian Calculation
#     H = np.zeros((len(prs), len(prs)))
#     for pr1 in range(len(prs)):
#         for pr2 in range(pr1, len(prs)):
#             H[pr1, pr2] = dt / refreshRate * np.sum(prsMat[:, pr1] * prsMat[:, pr2] * lambda_pred) + L2pen * 2 * (pr1 == pr2)
#             H[pr2, pr1] = H[pr1, pr2]  # Symmetric Hessian
#     return negloglike, dL, H

def negloglike_glm_basis(prs, NL, xconvki, yconvhi, y, dt, refreshRate, L2pen=0):
    nkbasis = xconvki.shape[1]  # Number of basis functions for k
    kprs = prs[:nkbasis]  # k basis functions
    hprs = prs[nkbasis:-1]  # h basis functions
    dc = prs[-1]  # DC term

    # Calculate xconvk_dc and yconvh
    xconvk_dc = np.dot(xconvki, kprs) + dc  # Shape (T,)
    yconvh = np.dot(yconvhi, hprs)  # Shape (T,)
    g = xconvk_dc + yconvh  # Shape (T,)

    # Apply the nonlinearity with clipping
    lambda_pred = NL(g)
    lambda_pred[np.isnan(lambda_pred)] = np.finfo(float).max  # Replace NaNs with large finite value

    # Calculate the negative log-likelihood
    negloglike = -np.dot(y, g) + dt * np.sum(lambda_pred) / refreshRate + L2pen * np.dot(prs, prs)
    print(f"Negative log-likelihood: {negloglike}")
    ## Gradient Calculation
    dL = np.zeros_like(prs)
    prsMat = np.hstack([xconvki, yconvhi, np.ones((xconvki.shape[0], 1))])  # Shape (T, nkbasis + nhbasis + 1)
    for pr in range(len(prs)):
        dL[pr] = -np.sum(prsMat[y.astype(bool), pr]) + dt / refreshRate * np.sum(prsMat[:, pr] * lambda_pred) + L2pen * 2 * prs[pr]

    ## Hessian Calculation
    H = np.zeros((len(prs), len(prs)))
    for pr1 in range(len(prs)):
        for pr2 in range(pr1, len(prs)):
            H[pr1, pr2] = dt / refreshRate * np.sum(prsMat[:, pr1] * prsMat[:, pr2] * lambda_pred) + L2pen * 2 * (pr1 == pr2)
            H[pr2, pr1] = H[pr1, pr2]  # Symmetric Hessian

    return negloglike, dL, H
