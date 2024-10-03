# %%
import numpy as np
import matplotlib.pyplot as plt
## From capturing the dynamical behavior of GLMS (Weber & Pillow 2017) MATLAB package

# %%
def generate_izhikevich_stim(cellType, T=10000):
    """
    This function generates a stimulus appropriate to the specified type of
    Izhikevich neuron. Defaults for each cell type are based on parameters 
    from "Capturing the dynamical repertoire of single neurons with GLMs" (Weber & Pillow 2017).

    Parameters:
    cellType: int, type of Izhikevich neuron. Choose from:
              1. tonic spiking
              2. phasic spiking
              3. tonic bursting
              4. phasic bursting
              5. mixed mode
              6. spike frequency adaptation
              7. Class 1
              8. Class 2
              9. spike latency
              11. resonator
              12. integrator
              13. rebound spike
              14. rebound burst
              15. threshold variability
              16. bistability
              18. accommodation
              19. inhibition-induced spiking
              20. inhibition-induced bursting
              21. bistability 2 (Not in original Izhikevich paper)
    T: float, max time of stimulus, in ms (default 10000)

    Returns:
    I: np.array, stimulus (current input)
    dt: float, time step in ms
    """
    
    # Check for unavailable behaviors
    if cellType in [10, 17]:
        print("This is a subthreshold behavior, which can't be captured by a GLM.")
        return None, None
    
    # Parameters (I and dt values for each cell type)
    pars = np.array([
        [14, 0.1],   # 1. tonic spiking
        [0.5, 0.1],  # 2. phasic spiking
        [10, 0.1],   # 3. tonic bursting
        [0.6, 0.1],  # 4. phasic bursting
        [10, 0.1],   # 5. mixed mode
        [20, 0.1],   # 6. spike frequency adaptation
        [25, 0.1],   # 7. Class 1
        [0.5, 0.1],  # 8. Class 2
        [3.49, 0.1], # 9. spike latency
        [0, 1],      # 10. subthreshold oscillations
        [0.3, 0.5],  # 11. resonator
        [27.4, 0.5], # 12. integrator
        [-5, 0.1],   # 13. rebound spike
        [-5, 0.1],   # 14. rebound burst
        [2.3, 1],    # 15. threshold variability
        [26.1, 0.05],# 16. bistability
        [0, 0.1],    # 17. depolarizing after-potential
        [20, 0.1],   # 18. accommodation
        [70, 0.1],   # 19. inhibition-induced spiking
        [70, 0.1],   # 20. inhibition-induced bursting
        [26.1, 0.05] # 21. bistability 2 (Not in original Izhikevich paper)
    ])
    
    Ival = pars[cellType-1, 0]
    dt = pars[cellType-1, 1]
    
    t = np.arange(dt, T + dt, dt)
    
    # Initialize stimulus
    I = np.zeros_like(t)
    stepLength = 500  # in ms
    nStepsUp = int(T / stepLength / 2)
    
    # Generate stimulus based on cell type
    if cellType in [1, 2, 3, 4, 5, 6, 19, 20]:
        if cellType in [19, 20]:
            I.fill(80)
        for i in range(nStepsUp):
            idx = (t > stepLength + stepLength * 2 * i) & (t < stepLength * 2 * (i + 1))
            I[idx] = Ival

    elif cellType in [7, 8]:
        if cellType == 7:
            stepSizes = np.arange(15, 31, 1)
        elif cellType == 8:
            stepSizes = np.arange(0.1, 0.7 + 0.025, 0.025)
        for i, stepSize in enumerate(stepSizes):
            idx = (t > stepLength + stepLength * 2 * i) & (t < stepLength * 2 * (i + 1))
            I[idx] = stepSize

    elif cellType == 9:
        stepLength = 150
        nStepsUp = int(T / stepLength / 2)
        for i in range(nStepsUp):
            idx = (t > stepLength * 1.94 + stepLength * 2 * i) & (t < stepLength * 2 * (i + 1))
            I[idx] = Ival

    elif cellType == 11:  # resonator
        stepLength = 150
        nStepsUp = int(T / stepLength / 2)
        for i in range(1, nStepsUp):
            pulseLength = int(5 / dt)
            idx1 = (t > stepLength + stepLength * 2 * i) & (t < stepLength + stepLength * 2 * i + pulseLength)
            idx2 = (t > stepLength + stepLength * 2 * i + pulseLength + 2 * i + pulseLength / 2) & \
                   (t < stepLength + stepLength * 2 * i + 2 * pulseLength + 2 * i + pulseLength / 2)
            I[idx1] = Ival
            I[idx2] = Ival

    elif cellType == 12:  # integrator
        stepLength = 250
        nStepsUp = int(T / stepLength / 2)
        for i in range(2, nStepsUp):
            pulseLength = int(4 / dt)
            idx1 = (t > stepLength + stepLength * 2 * i) & (t < stepLength + stepLength * 2 * i + pulseLength)
            idx2 = (t > stepLength + stepLength * 2 * i + pulseLength + 6 * i + pulseLength / 2) & \
                   (t < stepLength + stepLength * 2 * i + 2 * pulseLength + 6 * i + pulseLength / 2)
            I[idx1] = Ival
            I[idx2] = Ival

    elif cellType in [13, 14]:  # rebound spike, rebound burst
        for i in range(nStepsUp):
            idx = (t > stepLength * 1.6 + stepLength * 2 * i) & (t < stepLength * 2 * (i + 1))
            I[idx] = Ival

    elif cellType == 15:  # threshold variability
        dur = int(1 / dt)
        for i in range(nStepsUp * 2):
            idx = np.arange(stepLength * i - dur, stepLength * i)
            I[idx] = Ival
            if i % 2 == 1:
                I[idx - 25] = -Ival

    elif cellType in [16, 21]:  # bistability
        pulsePolarity = 1 if cellType == 16 else -1
        stepLength = 50
        nStepsUp = int(T / stepLength)
        I -= 65
        pulseDir = 2
        delay = -3
        for i in range(nStepsUp):
            if i % 2 == 1:
                idx = (t > stepLength + stepLength * i) & (t < stepLength + stepLength * i + pulseDir)
                I[idx] = Ival
            else:
                idx = (t > delay + stepLength + stepLength * i) & (t < delay + stepLength + stepLength * i + pulseDir)
                I[idx] = Ival * pulsePolarity

    elif cellType == 18:  # accommodation
        baseline = -70
        I.fill(baseline)
        for i in range(nStepsUp):
            if i % 2 == 1:
                idx = (t > stepLength + stepLength * 2 * i) & (t < stepLength * 2 * (i + 1))
                I[idx] = np.linspace(baseline, baseline + Ival, len(I[idx]))
            else:
                idx = (t > stepLength * 1.9 + stepLength * 2 * i) & (t < stepLength * 2 * (i + 1))
                I[idx] = np.linspace(baseline, baseline + Ival, len(I[idx]))

    return I, dt

def simulate_izhikevich(cellType, I, dt, jitter=0, plotFlag=True, saveFlag=False, fid=None):
    """
    Simulates an Izhikevich neuron of a specified type with given stimulus (current input).

    Parameters:
    - cellType: int, type of Izhikevich neuron (1 to 21 as described in the MATLAB function).
    - I: np.array, stimulus (current input).
    - dt: float, time step in ms.
    - jitter: float, add jitter to spike times, uniformly distributed over [-jitter, jitter] (in ms).
    - plotFlag: bool, whether to plot the simulation results.
    - saveFlag: bool, whether to save the simulation results to a file.
    - fid: str, file identifier (root directory for saving data, if saveFlag is set).
    
    Returns:
    - v: np.array, voltage response of the neuron.
    - u: np.array, membrane recovery variable.
    - spikes: np.array, vector of 0s and 1s indicating spikes.
    - cid: string, identifies cell type.
    """

    # Izhikevich parameters (a, b, c, d) for different neuron types
    pars = np.array([
        [0.02, 0.2, -65, 6],    # 1. tonic spiking
        [0.02, 0.25, -65, 6],   # 2. phasic spiking
        [0.02, 0.2, -50, 2],    # 3. tonic bursting
        [0.02, 0.25, -55, 0.05],# 4. phasic bursting
        [0.02, 0.2, -55, 4],    # 5. mixed mode
        [0.01, 0.2, -65, 5],    # 6. spike frequency adaptation
        [0.02, -0.1, -55, 6],   # 7. Class 1
        [0.2, 0.26, -65, 0],    # 8. Class 2
        [0.02, 0.2, -65, 6],    # 9. spike latency
        [0.05, 0.26, -60, 0],   # 10. subthreshold oscillations
        [0.1, 0.26, -60, -1],   # 11. resonator
        [0.02, -0.1, -55, 6],   # 12. integrator
        [0.03, 0.25, -60, 4],   # 13. rebound spike
        [0.03, 0.25, -52, 0],   # 14. rebound burst
        [0.03, 0.25, -60, 4],   # 15. threshold variability
        [1, 1.5, -60, 0],       # 16. bistability
        [1, 0.2, -60, -21],     # 17. depolarizing after-potential
        [0.02, 1, -55, 4],      # 18. accomodation
        [-0.02, -1, -60, 8],    # 19. inhibition-induced spiking
        [-0.026, -1, -45, 0],   # 20. inhibition-induced bursting
        [1, 1.5, -60, 0]        # 21. bistability 2
    ])

    # Cell IDs corresponding to different types
    cids = ['RS', 'PS', 'TB', 'PB', 'MM', 'FA', 'E1', 'E2', 'SL', 'SO', 'R', 'I', 'ES', 'EB', 'TV', 'B', 'DA', 'A', 'IS', 'IB', 'B2']

    a, b, c, d = pars[cellType-1]
    cid = cids[cellType-1]

    # Time array
    T = len(I) * dt
    t = np.arange(0, T, dt)

    # Initialize variables
    v = np.zeros_like(t)
    u = np.zeros_like(t)
    spikes = np.zeros_like(t)

    threshold = 30

    # Initial values based on cell type
    if cellType in [16, 21]:  # bistable
        v[0] = -54
        u[0] = -77
    elif cellType == 12:  # integrator
        v[0] = -90
        u[0] = 0
    elif cellType in [19, 20]:  # inhibition-induced spiking/bursting
        v[0] = -100
        u[0] = 80
    else:
        v[0] = -70
        u[0] = -14

    # Modify stimulus for bistable behavior
    if cellType == 21:
        I = np.abs(I + 65) - 65

    # Run the simulation
    for tt in range(len(I) - 1):
        dvdt = 0.04 * v[tt]**2 + 5 * v[tt] + 140 - u[tt] + I[tt]
        v[tt + 1] = v[tt] + dvdt * dt

        dudt = a * (b * v[tt + 1] - u[tt])
        u[tt + 1] = u[tt] + dudt * dt

        if v[tt + 1] > threshold:
            v[tt] = threshold  # Spike with uniform height
            v[tt + 1] = c
            u[tt + 1] += d
            spikes[tt + 1] = 1

    # Add jitter to spike times if needed
    if jitter > 0:
        spike_idx = np.where(spikes)[0]
        jitters = np.random.randint(-jitter // dt, jitter // dt + 1, size=spike_idx.shape)
        spike_idx = np.clip(spike_idx + jitters, 0, len(spikes) - 1)
        spikes[:] = 0
        spikes[spike_idx] = 1

    # Plot the results
    if plotFlag:
        sTimes = t[spikes.astype(bool)]

        plt.figure(figsize=(10, 6))

        # Plot current
        plt.subplot(2, 1, 1)
        plt.plot(t, I, label='Current Input')
        plt.ylabel('Current (I)')
        plt.title('Stimulus')
        plt.grid(True)

        # Plot voltage response
        plt.subplot(2, 1, 2)
        plt.plot(t, v, label='Voltage (v)')
        for s in sTimes:
            plt.axvline(x=s, color='k', linestyle='--', ymin=0.95, ymax=1.1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (v)')
        plt.title('simulated response')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # Save the data if required
    if saveFlag and fid:
        data = {'cellType': cellType, 'cid': cid, 'dt': dt, 'T': T, 'a': a, 'b': b, 'c': c, 'd': d,
                'threshold': threshold, 'I': I, 'u': u, 'v': v, 'spikes': spikes}
        np.savez(f"{fid}/{cid}_izhikevich_data.npz", **data)
        print(f"Saved: {fid}/{cid}_izhikevich_data.npz")

    return v, u, spikes, cid



