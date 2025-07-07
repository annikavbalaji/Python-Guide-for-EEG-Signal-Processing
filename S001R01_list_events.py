import mne

raw = mne.io.read_raw_fif('S001R01_preprocessed_raw.fif', preload=False)

# Print all annotation descriptions and event_id mapping
try:
    events, event_id = mne.events_from_annotations(raw)
    print('Event IDs:', event_id)
    print('Number of events:', len(events))
    print('All annotation descriptions:')
    for desc, code in event_id.items():
        print(f'  {desc!r}: {code}')
except Exception as e:
    print('Could not extract events:', e)
    if hasattr(raw, 'annotations'):
        print('Raw annotations:', raw.annotations)
