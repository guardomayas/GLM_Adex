from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
def fit_GLM(x,y,dt,kbasprs,ihbasprs,prs=None,nkt=100,softRect=0,plotFlag=True,maxIter=1000,tolFun=1e-12,L2pen=0):
    from utilities.BasisFunctions import makeBasis_StimKernel,makeBasis_PostSpike, sameconv
    from utilities.negloglike_glm import negloglike_glm_basis
    """
    This code fits a Poisson GLM to given data, using basis vectors to
    characterize the stimulus and post-spike filters.

    The inputs are:
     x: stimulus
     y: spiking data, vector of 0s and 1s
     dt: time step of x and y in ms
     nkt: number of ms in stimulus filter
     kbasprs: structure containing parameters of stimulus filter basis vectors
         kbasprs.neye: number of "identity" basis vectors near time of spike
         kbasprs.ncos: number of raised-cosine vectors to use
         kbasprs.kpeaks: position of first and last bump (realtive to identity bumps)
         kbasprs.b: how nonlinear to make spacings (larger -> more linear)
     ihbasprs: structure containing parameters of post-spike filter basis vectors
         ihbasprs.ncols: number of basis vectors for post-spike kernel
         ihbasprs.hpeaks: peak location for first and last vectors 
         ihbasprs.b: how nonlinear to make spacings (larger -> more linear)
         ihbasprs.absref: absolute refractory period, in ms

    The outputs are:
     The outputs are:
     k: stimulus filter
     h: post-spike filter
     dc: DC offset
     prs: full set of coefficients for basis vectors, [k_coeffs h_coeffs dc]
     kbasis: basis vectors for stimulus filter
     hbasis: basis vectors for post-spike filters    
    """
    refreshRate = 1000/dt # stimulus in ms, sampled at dt #Have to change when using light stimulus
    

    #Stimulus basis vectors
    kbasisTemp, kbasis = makeBasis_StimKernel(nkt, kbasprs=kbasprs)
    nkb = kbasis.shape[1]
    lenkb = kbasis.shape[0]
    kbasis = np.zeros((int(lenkb / dt), nkb))
    # Perform interpolation for each column
    for bNum in range(nkb):
        original_time_points = np.arange(1, lenkb + 1)
        new_time_points = np.linspace(1, lenkb, int(lenkb / dt))  # New time points with lenkb/dt samples
        kbasis[:, bNum] = np.interp(new_time_points, original_time_points, kbasisTemp[:, bNum])


    #Post-spike basis vectors
    ht,hbas,hbasis = makeBasis_PostSpike(ihbasprs,dt)
    hbasis = np.vstack([np.zeros((1, ihbasprs['ncols'])), hbasis])
    nkbasis = kbasis.shape[1] #Number of basis functions for K
    nhbasis = hbasis.shape[1]  #Number of basis functions for h

    if prs is None or len(prs) == 0:
        prs = np.random.randn(nkbasis + nhbasis + 1) * 0.01  #Revisar que tan importante es iniciarlos en 0 o cerca de 0. 
        # prs = np.zeros(nkbasis + nhbasis + 1)  # Initialize parameters

    ## Let's convolve the basis functions with the stimulus
    # convolution matrices
    xconvki = np.zeros((len(y), nkbasis))  # Stimulus convolved with stimulus filters. X and Y have same length 
    yconvhi = np.zeros((len(y), nhbasis))  # Spike response convolved with post-spike filters

       
    for knum in range(nkbasis):
        xconvki[:, knum] = sameconv(x, kbasis[:, knum])

    # Convolve the response with each flipped post-spike basis vector (for hnum in nhbasis)
    for hnum in range(nhbasis):
        yconvhi[:, hnum] = sameconv(y, np.flipud(hbasis[:, hnum])) #revisar que convulucion hacer. Ahorita es FFT. 

    
    # print("xconvki sample (first 5 rows):\n", xconvki[:5, :])
    # print("xconvki sample (last 5 rows):\n", xconvki[-5:, :])
    # print("yconvhi sample (first 5 rows):\n", yconvhi[:5, :])
    # print("yconvhi sample (last 5 rows):\n", yconvhi[-5:, :])
 
    if softRect == 0:
        def NL(g, clip_val=300): #no estoy seguro. 300 parece funcionar y parece un techo biologico. 
            return np.exp(np.clip(g, -clip_val, clip_val))
    else:
        NL = lambda g: np.log(1 + np.exp(g))  # Soft-rectified funcion
    
    ## LL minimization
    def objective(prs):
        negloglike, dL, H = negloglike_glm_basis(prs, NL, xconvki, yconvhi, y, dt, refreshRate, L2pen)
        print(f"Negative Log-Likelihood: {negloglike}")
        print(f"Gradient Sample (first 5): {dL[:5]}")
        return negloglike, dL

    # Wrapper for scipy.optimize to only return the objective value
    def objective_for_minimize(prs):
        negloglike, _ = objective(prs)
        return negloglike

    def gradient_for_minimize(prs):
        _, dL = objective(prs)
        return dL
    
    ## Parameter optimization
    result = minimize(objective_for_minimize, prs, jac=gradient_for_minimize, method='L-BFGS-B', 
                      options={'maxiter': maxIter, 'BFGS ': tolFun})

    # Extract optimized parameters
    prs_opt = result.x


    ##Calculate filters
    kprs = prs_opt[:nkbasis]  
    hprs = prs_opt[nkbasis:-1]  
    dc = prs_opt[-1]  # Optimized DC term
    print('Optimized parameters: ', prs_opt)
    # Calculate the actual stimulus filter and post-spike filter
    k = np.dot(kbasis, kprs) # k basis functions weighted by given parameters
    h = np.dot(hbasis, hprs)

    
    if plotFlag:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot the kbasis (stimulus basis vectors)
        axs[0, 0].set_title('Stimulus Basis Functions')
        for i in range(kbasis.shape[1]):
            axs[0, 0].plot(kbasis[:, i])
        axs[0, 0].set_xlim([0, len(k)])
        axs[0, 0].set_xticks(np.linspace(0, len(k), 5))
        axs[0, 0].set_xticklabels(np.round(np.linspace(0, len(k) * dt, 5)[::-1]).astype(int))

        # Plot the stimulus filter (k)
        axs[1, 0].set_title('Stimulus Filter (k)')
        axs[1, 0].plot(k)
        axs[1, 0].set_xlim([0, len(k)])
        axs[1, 0].set_xticks(np.linspace(0, len(k), 5))
        axs[1, 0].set_xticklabels(np.round(np.linspace(0, len(k) * dt, 5)[::-1]).astype(int))
        axs[1, 0].set_xlabel('Time (ms)')

        # Plot the hbasis (post-spike basis vectors)
        axs[0, 1].set_title('Post-Spike Basis Functions')
        for i in range(hbasis.shape[1]):
            axs[0, 1].plot(hbasis[:, i])
        axs[0, 1].set_xlim([0, len(h)])
        axs[0, 1].set_xticks(np.linspace(0, len(h), 5))
        axs[0, 1].set_xticklabels(np.round(np.linspace(0, len(h) * dt, 5)).astype(int))

        # Plot the post-spike filter (h)
        axs[1, 1].set_title('Post-Spike Filter (h)')
        axs[1, 1].plot(h)
        axs[1, 1].set_xlim([0, len(h)])
        axs[1, 1].set_xticks(np.linspace(0, len(h), 5))
        axs[1, 1].set_xticklabels(np.round(np.linspace(0, len(h) * dt, 5)).astype(int))
        axs[1, 1].set_xlabel('Time (ms)')

        plt.tight_layout()
        plt.show()

    return k, h, dc, prs_opt, kbasis, hbasis