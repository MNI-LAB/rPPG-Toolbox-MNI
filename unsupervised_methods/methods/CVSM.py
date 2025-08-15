"""CVSM (Classical Video-based Signal Measurement)
Depth compensation is the only thing that is done here.
"""

import numpy as np
def Depth_compensation(I_raw, Depth, timeWindow, Fs):
    I_comp = np.ones_like(I_raw, dtype=np.float32)
    best = 1
    best_rem = 1
    
    for _ in range(1):
        I_comp_ROI = np.ones(len(I_raw), dtype=np.float32)
        i = 1
        while (i * (timeWindow * Fs)) <= len(I_raw):
            cor = 2
            start_idx = (i - 1) * (timeWindow * Fs)
            end_idx = i * (timeWindow * Fs)
            I_seg = I_raw[start_idx:end_idx]
            D_seg = Depth[start_idx:end_idx]
            
            if np.std(D_seg) == 0:
                bI_comp = (I_seg - np.mean(I_seg)) / (np.std(I_seg) + 1e-7) if np.std(I_seg) > 0 else np.zeros_like(I_seg)
                I_comp_ROI[start_idx:end_idx] = bI_comp
                i += 1
                continue
            
            for bi in np.arange(0.2, 5.01, 0.01):
                bI_comp_raw = I_seg / (D_seg ** (-bi))
                corr_v = np.corrcoef(bI_comp_raw, D_seg)
                corr_ = abs(corr_v[1, 0])
                if corr_ < cor:
                    cor = corr_
                    best = bI_comp_raw
            I_comp_ROI[start_idx:end_idx] = (best - np.mean(best)) / (np.std(best) + 1e-7) if np.std(best) > 0 else np.zeros_like(best)
            i += 1
        
        start_idx = ((i - 1) * (timeWindow * Fs))
        if start_idx < len(I_raw):
            cor = 2
            I_rem = I_raw[start_idx:]
            D_rem = Depth[start_idx:]

            if np.std(D_rem) == 0:
                bI_comp = (I_rem - np.mean(I_rem)) / (np.std(I_rem) + 1e-7) if np.std(I_rem) > 0 else np.zeros_like(I_rem)
                I_comp_ROI[start_idx:len(I_raw)] = bI_comp
            else:
                for bii in np.arange(0.2, 5.1, 0.1):
                    bI_comp_raw = I_rem / (D_rem ** (-bii))
                    corr_v = np.corrcoef(bI_comp_raw, D_rem)
                    corr_ = abs(corr_v[1, 0])
                    if corr_ < cor:
                        cor = corr_
                        best_rem = bI_comp_raw
                I_comp_ROI[start_idx:len(I_raw)] = (best_rem - np.mean(best_rem)) / (np.std(best_rem) + 1e-7) if np.std(best_rem) > 0 else np.zeros_like(best_rem)
        I_comp = I_comp_ROI
    return I_comp