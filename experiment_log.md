# Experiments Done and parameters used
## OMNICAN (RCNN + TSCAN) August 13, 2025
- MY_FFT HR calculation
- rPPG-Toolbox Bandpass filter
### Result
FFT MAE (FFT Label): 1.1111111111111118 +/- 0.3959425806109992
FFT RMSE (FFT Label): 3.3333333333333353 +/- 1.9898305973398829
FFT MAPE (FFT Label): 1.379440665154952 +/- 0.4946561683586922
FFT Pearson (FFT Label): 0.9256268798865707 +/- 0.048453946231794574
FFT SNR (FFT Label): 0.7607726187205845 +/- 0.5482972857583618 (dB)
FFT Percentage Accuracy (within 10%): 88.9% (56/63 measurements) +/- 7.9%
## CVSM (Classical Method) August 14, 2025
- MY_FFT HR Calculation
- Savgol filter
### Result
FFT MAE (FFT Label): 2.4138609230391745 +/- 0.9764991760091088
FFT RMSE (FFT Label): 8.058961737225001 +/- 5.761310808031412
FFT MAPE (FFT Label): 2.78244176527215 +/- 1.0960848545217872
FFT Pearson (FFT Label): 0.4740775991291359 +/- 0.11366987507719846
FFT SNR (FFT Label): 0.7411545687353587 +/- 0.35059365589023955 (dB)
FFT Percentage Accuracy (within 10%): 93.5% (58/62 measurements) +/- 6.5%
## OMNiCAN August 15, 2025
- Do chunk, 145 each chunk, total frames always truncate to 580.
- MY_FFT HR calculation
- Bandpass filter
- log folder path: /nfs/turbo/coe-mni/toolbox_runs/omnican_gd_tvt_exp_do_chunk
## OMNICAN September 21, 2025
- Chunks of 200, @20fps
- Using Diff normalized NIR + Depth channels
