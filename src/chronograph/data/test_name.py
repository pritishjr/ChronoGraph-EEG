import mne, numpy as np, re

CANON = ['FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3','C3-P3','P3-O1',
         'FP2-F4','F4-C4','C4-P4','P4-O2','FP2-F8','F8-T8','T8-P8','P8-O2',
         'FZ-CZ','CZ-PZ']
OLD2NEW = {'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}

def norm(name):
    name = re.sub(r'-\d+$', '', name.strip().upper())
    return '-'.join(OLD2NEW.get(p, p) for p in name.split('-'))

def load_canonical(path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose='error')
    assert int(raw.info['sfreq']) == 256
    idx = {}
    for i, n in enumerate(raw.ch_names):
        k = norm(n)
        if k in CANON and k not in idx:
            idx[k] = i
    data = raw.get_data()
    x = np.zeros((len(CANON), data.shape[1]), np.float32)
    mask = np.zeros(len(CANON), bool)
    for j, c in enumerate(CANON):
        if c in idx:
            x[j], mask[j] = data[idx[c]], True
    return x, mask, [c for c in CANON if c not in idx]

def main():
    
    #dataset path:
    path = "data/raw/physionet.org/files/chbmit/1.0.0"
    
    #

if __name__ == "__main__":
    main()