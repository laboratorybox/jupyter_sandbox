import json
import numpy as np
import hither2 as hi
import kachery as ka

@hi.function('compute_units_info', version='0.1.1')
@hi.container('docker://magland/spikeforest_sandbox:0.1.0')
@hi.local_modules(['/workspaces/spikeforest2/spikeforest2_utils'])
def compute_units_info(recording_path, sorting_path):
    from spikeforest2_utils import AutoRecordingExtractor, AutoSortingExtractor
    
    # we should not need to do this config (think about it)
    with ka.config(fr='default_readonly'):
        recording = AutoRecordingExtractor(recording_path)
        sorting = AutoSortingExtractor(sorting_path, samplerate=recording.get_sampling_frequency())
    obj = _compute_units_info(recording=recording, sorting=sorting)
    return obj

def _compute_units_info(*, recording, sorting, channel_ids=[], unit_ids=[]):
    import spikeextractors as se
    import spiketoolkit as st

    if (channel_ids) and (len(channel_ids) > 0):
        recording = se.SubRecordingExtractor(parent_recording=recording, channel_ids=channel_ids)

    # load into memory
    print('Loading recording into RAM...')
    recording = se.NumpyRecordingExtractor(timeseries=recording.get_traces(), sampling_frequency=recording.get_sampling_frequency())

    # do filtering
    print('Filtering...')
    recording = st.preprocessing.bandpass_filter(recording=recording, freq_min=300, freq_max=6000)
    recording = se.NumpyRecordingExtractor(timeseries=recording.get_traces(), sampling_frequency=recording.get_sampling_frequency())

    if (not unit_ids) or (len(unit_ids) == 0):
        unit_ids = sorting.get_unit_ids()

    print('Computing channel noise levels...')
    channel_noise_levels = _compute_channel_noise_levels(recording=recording)

    # No longer use subset to compute the templates
    print('Computing unit templates...')
    templates = _compute_unit_templates(recording=recording, sorting=sorting, unit_ids=unit_ids, max_num=100)

    print(recording.get_channel_ids())

    ret = []
    for i, unit_id in enumerate(unit_ids):
        print('Unit {} of {} (id={})'.format(i + 1, len(unit_ids), unit_id))
        template = templates[i]
        max_p2p_amps_on_channels = np.max(template, axis=1) - np.min(template, axis=1)
        peak_channel_index = np.argmax(max_p2p_amps_on_channels)
        peak_channel = recording.get_channel_ids()[peak_channel_index]
        peak_signal = np.max(np.abs(template[peak_channel_index, :]))
        times0 = sorting.get_unit_spike_train(unit_id=unit_id)
        times0_sec = np.array(times0) / recording.get_sampling_frequency()
        isi_violation_rate, _ = _isi_violations(times0_sec, 0, recording.get_num_frames() / recording.get_sampling_frequency(), isi_threshold=2.5 / 1000)
        info0 = dict()
        info0['unit_id'] = int(unit_id)
        info0['snr'] = peak_signal / channel_noise_levels[peak_channel_index]
        info0['peak_channel'] = int(recording.get_channel_ids()[peak_channel])
        train = sorting.get_unit_spike_train(unit_id=unit_id)
        info0['num_events'] = int(len(train))
        info0['firing_rate'] = float(len(train) / (recording.get_num_frames() / recording.get_sampling_frequency()))
        info0['isi_violation_rate'] = isi_violation_rate
        ret.append(info0)
    return ret

def _compute_channel_noise_levels(recording):
    channel_ids = recording.get_channel_ids()
    # M=len(channel_ids)
    samplerate = int(recording.get_sampling_frequency())
    X = recording.get_traces(start_frame=samplerate * 1, end_frame=samplerate * 2)
    ret = []
    for ii in range(len(channel_ids)):
        # noise_level=np.std(X[ii,:])
        noise_level = np.median(np.abs(X[ii, :])) / 0.6745  # median absolute deviation (MAD)
        ret.append(noise_level)
    return ret

def _compute_unit_templates(*, recording, sorting, unit_ids, snippet_len=50, max_num=100, channels=None):
    ret = []
    for unit in unit_ids:
        # print('Unit {} of {}'.format(unit,len(unit_ids)))
        waveforms = _get_random_spike_waveforms(recording=recording, sorting=sorting, unit=unit, snippet_len=snippet_len, max_num=max_num, channels=None)
        template = np.median(waveforms, axis=2)
        ret.append(template)
    return ret

def _get_random_spike_waveforms(*, recording, sorting, unit, snippet_len, max_num, channels=None):
    st = sorting.get_unit_spike_train(unit_id=unit)
    num_events = len(st)
    if num_events > max_num:
        event_indices = np.random.choice(range(num_events), size=max_num, replace=False)
    else:
        event_indices = range(num_events)

    spikes = recording.get_snippets(reference_frames=st[event_indices].astype(int), snippet_len=snippet_len, channel_ids=channels)
    if len(spikes) > 0:
        spikes = np.dstack(tuple(spikes))
    else:
        spikes = np.zeros((recording.get_num_channels(), snippet_len, 0))
    return spikes

def _isi_violations(spike_train, min_time, max_time, isi_threshold, min_isi=None):
    """Calculate Inter-Spike Interval (ISI) violations for a spike train.

    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    Originally written in Matlab by Nick Steinmetz (https://github.com/cortex-lab/sortingQuality)
    Converted to Python by Daniel Denman

    Inputs:
    -------
    spike_train : array of monotonically increasing spike times (in seconds) [t1, t2, t3, ...]
    min_time : start of recording (in seconds)
    max_time : end of recording (in seconds)
    isi_threshold : threshold for classifying adjacent spikes as an ISI violation
      - this is the biophysical refractory period
    min_isi : minimum possible inter-spike interval (default = None)
      - this is the artificial refractory period enforced by the data acquisition system
        or post-processing algorithms
      - if set to None, all spikes will be included

    Outputs:
    --------
    fpRate : rate of contaminating spikes as a fraction of overall rate
      - higher values indicate more contamination
    num_violations : total number of violations detected

    """

    isis = np.diff(spike_train)

    if min_isi is not None:
        duplicate_spikes = np.where(isis <= min_isi)[0]
        spike_train = np.delete(spike_train, duplicate_spikes + 1)
    else:
        min_isi = 0

    num_spikes = len(spike_train)
    num_violations = sum(isis < isi_threshold)
    violation_time = 2 * num_spikes * (isi_threshold - min_isi)
    total_rate = _firing_rate(spike_train, min_time, max_time)
    violation_rate = num_violations / violation_time
    fpRate = violation_rate / total_rate

    return fpRate, num_violations

def _firing_rate(spike_train, min_time=None, max_time=None):
    """Calculate firing rate for a spike train.

    If either temporal bound is not specified, the first and last spike time are used by default.

    Inputs:
    -------
    spike_train : array of spike times (in seconds)
    min_time : time of first possible spike (optional)
    max_time : time of last possible spike (optional)

    Outputs:
    --------
    fr : float
        Firing rate in Hz

    """

    if min_time is None:
        min_time = np.min(spike_train)

    if max_time is None:
        max_time = np.max(spike_train)

    duration = max_time - min_time
    fr = spike_train.size / duration

    return fr