
import numpy as np
def makeBasis_StimKernel(nkt, kbasprs):
    
    """
    %  Generates a basis consisting of raised cosines and several columns of
    %  identity matrix vectors for temporal structure of stimulus kernel
    %
    %  Args: kbasprs = dictionary with fields: 
    %          neye = number of identity basis vectors at front
    %          ncos = # of vectors that are raised cosines
    %          kpeaks = 2-vector, with peak position of 1st and last vector,
    %             relative to start of cosine basis vectors (e.g. [0 10])
    %          b = offset for nonlinear scaling.  larger values -> more linear
    %             scaling of vectors.  bk must be >= 0
    %        nkt = number of time samples in basis (optional)
    %
    %  Output:
    %        kbas = orthogonal basis
    %        kbasis = standard (non-orth) basis. Not currently implemented. Both are standard.
    """
    neye = kbasprs['neye']
    ncos = kbasprs['ncos']
    kpeaks = kbasprs['kpeaks']
    b = kbasprs['b']
    
    kdt = 1 #sapcing of X axis must be in units (of 1)
    
    ### Non linearity for stretching x and its inverse
    #The nonlinearity is used to strecth the xaxis (time) before creating the raised cosine basis functions. 
    #This is key: it allos the basis function be spaced more densely in some regions and more sparse in others
    #We can modify this 

    nlin = lambda x: np.log(x+1e-20)  # Nonlinear transformation (log scale)
    invnl = lambda x: np.exp(x)-1e-20
    
    #raised cosine basis
    yrnge = nlin(np.array(kpeaks) + b)
    db = np.diff(yrnge)[0] / (ncos - 1)  # Spacing between raised cosine peaks
    ctrs = np.arange(yrnge[0], yrnge[1] + db, db)[:ncos]  # Centers for basis vectors, restricted to ncos
    mxt = invnl(yrnge[1] + 2 * db) - b  # Maximum time bin


    kt0 = np.arange(0, mxt + kdt-1, kdt)  # Time samples
    nt = len(kt0)

    # the actual raised cosine basis
    ff = lambda x, c, dc: (np.cos(np.clip((x - c) * np.pi / dc / 2, -np.pi, np.pi)) + 1) / 2
    kbasis0 = np.array([ff(nlin(kt0 + b), ctr, db) for ctr in ctrs]).T  # basis vectors

    # Concatenate identity-vectors
    nkt0 = len(kt0)
    kbasis = np.hstack([np.vstack([np.eye(neye), np.zeros((nkt0, neye))]),
                        np.vstack([np.zeros((neye, ncos)), kbasis0])])
    kbasis = np.flipud(kbasis)  # Flip so fine timescales are at the end
    
    # Adjust to have the correct number of time samples (nkt)
    nkt0 = kbasis.shape[0]
    if nkt is not None:
        if nkt0 < nkt:
            # Pad with zeros if kbasis has fewer rows than nkt
            kbasis = np.vstack([np.zeros((nkt - nkt0, neye + ncos)), kbasis])
        elif nkt0 > nkt:
            kbasis = kbasis[-nkt:, :]

    # Normalize columns 
    kbasis = kbasis / np.sqrt(np.sum(kbasis ** 2, axis=0))

    kbas = kbasis
    return kbas,kbasis 

def makeBasis_PostSpike(ihprs,dt,iht0=None):
    """
    % Make nonlinearly stretched basis consisting of raised cosines
    % Inputs: 
    %     prs = param structure with fields:
    %            ncols = # of basis vectors
    %            hpeaks = 2-vector containg [1st_peak  last_peak], the peak 
    %                      location of first and last raised cosine basis vectors
    %            b = offset for nonlinear stretching of x axis:  y = log(x+b) 
    %                 (larger b -> more nearly linear stretching)
    %            absref = absolute refractory period (optional)
    %     dt = grid of time points for representing basis
    %     iht (optional) = cut off time (or extend) basis so it matches this
    %
    %  Outputs:  iht = time lattice on which basis is defined
    %            ihbas = orthogonalized basis
    %            ihbasis = original (non-orthogonal) basis 
    %
    %  Example call:
    %
    %  ihbasprs.ncols = 5;  
    %  ihbasprs.hpeaks = [.1 2];  
    %  ihbasprs.b = .5;  
    %  ihbasprs.absref = .1;  %% (optional)
    """
    ncols = ihprs['ncols']
    # print(ncols)
    hpeaks = ihprs['hpeaks']
    b = ihprs['b']
    absref = ihprs.get('absref', 0)  # Optional: Default is 0

    if (hpeaks[0]+b) < 0:
        print('ERROR: b + first peak location: must be greater than 0')

    if absref >= dt: # use one less basis vector 
        ncols -=  1
    elif absref > 0:
        print('WARNING: Refractory period is too small for time bin sizes')
    
    nlin = lambda x: np.log(x+1e-20)  # Nonlinear transformation (log scale)
    invnl = lambda x: np.exp(x)-1e-20
    # print(ncols)
    #raised cosine basis
    yrnge = nlin(np.array(hpeaks) + b)
    db = np.diff(yrnge)[0] / (ncols - 1)  # Spacing between raised cosine peaks
    ctrs = np.arange(yrnge[0], yrnge[1] + db, db)[:ncols]  # Centers for basis vectors, restricted to ncos
    mxt = invnl(yrnge[1] + 2 * db) - b  # Maximum time bin
    # print(ctrs)
    iht = np.arange(0,mxt,dt) # time lattice on which basis is defined
    nt = len(iht) #number of points in iht

    #cosine basis function
    ff = lambda x, c, dc: (np.cos(np.clip((x - c) * np.pi / dc / 2, -np.pi, np.pi)) + 1) / 2
    ihbasis = np.array([ff(nlin(iht + b), ctr, db) for ctr in ctrs]).T  # basis vectors

    # Set first basis vector bins before 1st peak to 1
    ihbasis[iht <= hpeaks[0], 0] = 1

    # Step-function for absolute refractory period
    if absref >= dt:
        ih0 = np.zeros_like(iht)
        ih0[iht < absref] = 1
        ihbasis[iht < absref, :] = 0
        ihbasis = np.column_stack((ih0, ihbasis))
    ihbas, _ = np.linalg.qr(ihbasis)  #orthogonal basis functions so they correlated not.
    
    # Handle optional time lattice iht0
    if iht0 is not None:
        if (iht0[1] - iht0[0]) != dt:
            raise ValueError('iht passed in has different time bin size')

        niht = len(iht0)
        if iht[-1] > iht0[-1]:  # Truncate basis
            iht = iht0
            ihbasis = ihbasis[:niht, :]
            ihbas = ihbas[:niht, :]
        elif iht[-1] < iht0[-1]:  # Extend basis
            nextra = niht - len(iht)
            iht = iht0
            ihbasis = np.vstack([ihbasis, np.zeros((nextra, ncols))])
            ihbas = np.vstack([ihbas, np.zeros((nextra, ncols))])

    return iht, ihbas, ihbasis


def sameconv(A, B):
    am = len(A)
    bm = len(B)
    print("A, B shape", A.shape, B.shape)
    nn = am + bm - 1  # Full length of convolution result
    G = np.fft.ifft(np.fft.fft(A, nn) * np.fft.fft(np.flip(B), nn)).real
    G = G[:am]
    
    return G

# def sameconv(A, B):
#     from scipy.signal import convolve
#     G = convolve(A, B[::-1], mode='same')
#     return G


# def sameconv(A, B):
#     am = len(A)
#     bm = len(B)
#     from scipy.signal import convolve
#     G = convolve(A, B[::-1], mode='same')

    
#     return G

