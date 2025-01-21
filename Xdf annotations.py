streams, header = pyxdf.load_xdf(recording_path / 'Realtime_annotations.xdf')
data = streams[1]["time_series"].Tsfreq = float(streams[1]["info"]["nominal_srate"][0])
info = mne.create_info(67, sfreq)
raw = mne.io.RawArray(data, info)
annotation_stream= streams[0]
haba=list(annotation_stream['time_stamps'])
baseline_timestamp = 512046.19183777
haba2 = [onset - baseline_timestamp for onset in haba]
descriptions = [event[0] for event in annotation_stream['time_series']]
annotations = mne.Annotations(onset=haba2, duration = 0, description=descriptions)
raw.set_annotations(annotations)
