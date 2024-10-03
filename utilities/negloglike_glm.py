import numpy as np
def negloglike_glm_basis(prs,NL,xconvki,yconvhi,y,dt,refreshRate,L2pen=0):
    """
    calculates negative log likelihood
    prs:        vector of parameters, coefficients for basis functions in order
%                     [kprs (stim filter); hprs (post-spike filter); dc]
%   NL:         function handle for nonlinearity
%   xconvki:    stimulus convolved with each filter basis vector,
%                     upsampled to match response sampling rate
%   yconvhi:    response vector convolved with each filter basis vector
%   y:          response vector (zeros and ones)
%   dt:         time scale of y (in frames/stimulus frame)
%   refreshRate: refresh rate of stimulus (frames/sec)
%   L2pen:      penalty on L2 norm of prs vector
    """
    nkbasis = xconvki.shape[1] # # Number of basis functions for stimulus filter
    
    # Extract weights for stimulus filter, post-spike filter, and DC term
    kprs = prs[:nkbasis]  # Stimulus weights
    hprs = prs[nkbasis:-1]  # Post-spike weights
    dc = prs[-1]  # DC bias term, spontaneuous firing rate?

    # Compute the linear combination of the stimulus and post-spike terms
    xconvk_dc = np.dot(xconvki, kprs) + dc
    yconvh = np.dot(yconvhi, hprs)
    g = xconvk_dc + yconvh  # Linear prediction

    # Apply the nonlinearity (e.g., exponential or soft rectification)
    lambda_pred = NL(g)
    print(refreshRate)
    # Compute negative log-likelihood
    negloglike = -np.dot(y, g) + dt * np.sum(lambda_pred) / refreshRate + L2pen * np.dot(prs, prs)
    
    print(f"Negative log-likelihood: {negloglike}")

    if not np.isfinite(negloglike):
        print("Warning: Non-finite negative log-likelihood encountered!")

  # Compute the gradient
    dL = np.zeros_like(prs)
    prsMat = np.hstack([xconvki, yconvhi, np.ones((xconvki.shape[0], 1))])
    dL = -np.sum(prsMat[y.astype(bool)], axis=0) + dt / refreshRate * np.sum(prsMat * lambda_pred[:, None], axis=0) + L2pen * 2 * prs
    
    # Compute the Hessian
    H = np.zeros((len(prs), len(prs)))
    for pr1 in range(len(prs)):
        for pr2 in range(pr1, len(prs)):
            H[pr1, pr2] = dt / refreshRate * np.sum(prsMat[:, pr1] * prsMat[:, pr2] * lambda_pred) + L2pen * 2 * (pr1 == pr2)
            H[pr2, pr1] = H[pr1, pr2]  # Symmetry of Hessian

    return negloglike, dL, H