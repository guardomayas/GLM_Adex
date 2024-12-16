import numpy as np
import datajoint as dj
import matplotlib.pyplot as plt

def getStim(dataset):
    # 1) create the random generator stream using the seed
    noiseGen = np.random.RandomState(seed=int(dataset.noise_seed))
    
    preFrames = int(dataset.frame_rate * dataset.pre_time / 1000)
    stimFrames = int(dataset.frame_rate * dataset.stim_time / 1000)
    tailFrames = int(dataset.frame_rate * dataset.tail_time / 1000)
    # nFrames = preFrames + stimFrames + tailFrames
    
    # 2) draw random values  and convert to gaussian
    nFrames = stimFrames // int(dataset.frame_dwell)
    randVals = noiseGen.rand(nFrames, 2)
    randVals = np.sqrt(-2*np.log(randVals[:,0])) * np.cos(2*np.pi*randVals[:,1])
    
    # 4) add on mean, std, pre/tail time, frame_dwell, etc.
    
    contrast = randVals * dataset.contrast # relative to spot mean level
    luminance = dataset.spot_mean_level * (1 + contrast)
    if dataset.mean_level:
        contrast_c = (luminance - dataset.mean_level) / dataset.mean_level # relative to the background level
        
        mean_c = (dataset.spot_mean_level - dataset.mean_level) / dataset.mean_level
        
                
        # NOTE:
        # Weber contrast:
        # contrast -> luminance =  mean + contrast * mean = (1+contrast) * mean
        # luminance -> contrast = (luminance - mean)/mean    
    else: # luminance mode
        contrast_c = luminance
        mean_c = dataset.spot_mean_level
    contrast_c = np.clip(contrast_c, -1, 1)
    print(contrast_c.max(), contrast_c.min())
    return np.concatenate((
        np.ones(preFrames) * mean_c,
        np.repeat(contrast_c, dataset.frame_dwell),
        np.ones(tailFrames) * mean_c,
    ))

def plot_spike_info(ds, bin_size=25/1000):
    trial = ds.iloc[0]
    pre = trial.pre_time/1000
    stim = trial.stim_time/1000
    tail = trial.tail_time/1000
    total = pre + stim + tail
    _,axs = plt.subplot_mosaic(
    '''
    ab
    c.
    ''', width_ratios=[5,1], height_ratios=[5,1])
    # Raster plot
    plt.sca(axs['a'])
    for i, trial in ds.iterrows():
        plt.vlines(ds['spike_times'][i], i - .5, i + .5)  
    plt.xlim([0, total])
    plt.xlabel('Time (s)')
    plt.ylabel('Trial Number')
    plt.title(f'Raster Plot. Effective frame rate: {trial['frame_rate']/trial['frame_dwell']} Hz')

    # Spike count per trial
    plt.sca(axs['b'])
    plt.plot(ds['spike_count'], np.arange(len(ds)), color='w')
    
    plt.xlabel('Spike Count')
    plt.title('Spike Count per Trial')
    axs['b'].yaxis.set_label_position("right")
    axs['b'].yaxis.tick_right()
    #PSTH
    plt.sca(axs['c'])
    duration = total 
    nbins = int(duration / bin_size)

    time_bins = np.linspace(0, duration, nbins + 1)

    all_spike_times = np.concatenate([np.array(trial['spike_times']).flatten() for _, trial in ds.iterrows()])

    psth, _ = np.histogram(all_spike_times, bins=time_bins)

    psth = psth / len(ds) / bin_size

    plt.bar(time_bins[:-1], psth, width=bin_size, align='edge', color='gray')
    plt.xlabel('Time (s)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title(f'PSTH. Total spikes')
    plt.xlim([0, total])
    plt.tight_layout()
    plt.show()
    
    # ISI Histogram 
    figure = plt.gcf()
    all_isis = []  
    for i, trial in ds.iterrows():
        spike_times = np.array(trial['spike_times']).flatten()*1000
        if len(spike_times) > 1:  # Ensure there is more than one spike
            isis = np.diff(spike_times)
            all_isis.extend(isis)

    plt.hist(all_isis, bins=100, color='blue', edgecolor='w')
    plt.xlabel('ISI (ms)')
    plt.ylabel('Count')
    plt.title('ISI Histogram')
    plt.tight_layout()


    plt.show()
    return time_bins, psth

def STA(df, window_size_ms, plotFlag = True):
    sta_window_samples = None  # Based on sampling rate
    sta_accumulator = None
    total_spikes = 0

    for _, trial in df.iterrows():
        stimulus = np.array(trial['stimulus'])  
        spike_indices = np.array(trial['spike_indices']).flatten() #units are samples (on amplifier)
        # samples / (samples/sec) * (frames/sec) = frames ~ frame during which spike occurred

        scale_factor = trial['frame_rate']/ trial['sample_rate'] 
        scaled_spike_indices = (spike_indices * scale_factor).astype(int) # frame index of spike

        if sta_window_samples is None: #number of frame samples in window
            sta_window_samples = int((window_size_ms / 1000) * trial['frame_rate']) 

        valid_spike_indices = scaled_spike_indices[scaled_spike_indices < len(stimulus)]

        for spike_idx in valid_spike_indices:
            if spike_idx >= sta_window_samples:  
                window = stimulus[spike_idx - sta_window_samples:spike_idx]
                if sta_accumulator is None:
                    sta_accumulator = np.zeros_like(window)  
                sta_accumulator += window
                total_spikes += 1

    print('Total spikes:', total_spikes)

    if total_spikes > 0 and sta_accumulator is not None:
        sta = sta_accumulator / total_spikes
        print('STA shape:', sta.shape)
        if plotFlag:
            time_axis = np.linspace(-window_size_ms, 0, sta_window_samples)
            plt.figure(figsize=(8, 4))
            plt.plot(time_axis, sta,'w')
            plt.xlabel('Time before spike (ms)')
            plt.ylabel('Average Stimulus')
            plt.title(f'Temporal STA with effective frame rate = {trial["frame_rate"]/trial["frame_dwell"]} Hz')
            plt.show()
    else:
        print("STA accumulator is empty or no valid spikes processed.")

    return sta, sta_window_samples #To return STA as a filter