from datetime import datetime   
import os
import glob
import json
import argparse
from typing import List, Tuple

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from neural_methods.model.OMNI_CAN import OMNICAN
from evaluation import post_process as pp
from dataset.data_loader.BaseLoader import BaseLoader
def face_crop(frame):
    detector = cv2.CascadeClassifier(
        './dataset/haarcascade_frontalface_default.xml')

    # Computed face_zone(s) are in the form [x_coord, y_coord, width, height]
    # (x,y) corresponds to the top-left corner of the zone to define using
    # the computed width and height.
    face_zone = detector.detectMultiScale(frame[:, :, :3].astype(np.uint8))

    if len(face_zone) < 1:
        print("ERROR: No Face Detected")
        face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
    elif len(face_zone) >= 2:
        # Find the index of the largest face zone
        # The face zones are boxes, so the width and height are the same
        max_width_index = np.argmax(face_zone[:, 2])  # Index of maximum width
        face_box_coor = face_zone[max_width_index]
        print("Warning: More than one faces are detected. Only cropping the biggest one.")
    else:
        face_box_coor = face_zone[0]   
    return face_box_coor

def _list_clip_dirs(root_dir: str, max_clips: int = 10) -> List[str]:
    candidates = sorted([p for p in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(p)])
    return candidates[:max_clips]


def _read_rgbd_frames(clip_dir: str, display_frames: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    intensity_paths = sorted(glob.glob(os.path.join(clip_dir, 'Intensity', '*.png')))
    depth_paths = sorted(glob.glob(os.path.join(clip_dir, 'Depth', '*.png')))
    num_frames = min(len(intensity_paths), len(depth_paths))
    print(f"Found {num_frames} frames in {clip_dir}")
    if num_frames == 0:
        return np.empty((0,)), np.empty((0,))

    intensity_list = []
    depth_list = []
    
    # Display every 10th frame or first few frames
    display_indices = list(range(0, min(10, num_frames), max(1, num_frames // 10)))
    
    for i in range(num_frames):
        intensity = cv2.imread(intensity_paths[i], cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED)
        if intensity is None or depth is None:
            continue
        if depth.ndim == 3:
            depth = depth[:, :, :1]
        intensity = intensity[:, :, 1:2]
        
        # Face cropping
        face_region = face_crop(intensity)
        intensity = intensity[max(face_region[1], 0):min(face_region[1] + face_region[3], intensity.shape[0]),
                    max(face_region[0], 0):min(face_region[0] + face_region[2], intensity.shape[1])]
        depth = depth[max(face_region[1], 0):min(face_region[1] + face_region[3], depth.shape[0]),
                max(face_region[0], 0):min(face_region[0] + face_region[2], depth.shape[1])]
        
        # Resize intensity to match depth spatial size if needed
        if intensity.shape[:2] != depth.shape[:2]:
            intensity = cv2.resize(intensity, (depth.shape[1], depth.shape[0]))
        intensity_list.append(intensity)
        depth_list.append(depth)
        
        # Display frames
        if display_frames and i in display_indices:
            # Convert intensity to displayable format (0-255)
            intensity_display = intensity.squeeze() if intensity.ndim == 3 else intensity
            if intensity_display.dtype != np.uint8:
                intensity_display = (intensity_display * 255).astype(np.uint8)
            
            # Resize for display if too large
            display_size = 400
            if intensity_display.shape[0] > display_size or intensity_display.shape[1] > display_size:
                scale = display_size / max(intensity_display.shape)
                new_h = int(intensity_display.shape[0] * scale)
                new_w = int(intensity_display.shape[1] * scale)
                intensity_display = cv2.resize(intensity_display, (new_w, new_h))
            
            cv2.imshow(f'Intensity Frame {i}', intensity_display)
            cv2.waitKey(100)  # Display for 100ms
            print(f"  Displayed intensity frame {i}: shape {intensity.shape}")

    if not intensity_list:
        return np.empty((0,)), np.empty((0,))

    intensity_arr = np.asarray(intensity_list)  # (T,H,W,1)
    depth_arr = np.asarray(depth_list)  # (T,H,W)
    print(f"Intensity shape: {intensity_arr.shape}, Depth shape: {depth_arr.shape}")
    
    if display_frames:
        cv2.destroyAllWindows()  # Close all display windows
        print("  Frame display completed")
    
    return intensity_arr, depth_arr


def _read_gt_ppg(clip_dir: str) -> np.ndarray:
    """Read ground truth PPG from JSON file in the same format as iPadDataLoader."""
    clip_name = os.path.basename(clip_dir)
    json_path = os.path.join(clip_dir, f"{clip_name}.json")
    
    if not os.path.exists(json_path):
        return np.empty((0,))
    
    try:
        with open(json_path, "r") as f:
            labels = json.load(f)
            waves = [label["waveform"] for label in labels]
        return np.asarray(waves)
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Warning: Could not load GT PPG from {json_path}: {e}")
        return np.empty((0,))


def _resize_and_standardize(intensity: np.ndarray, depth: np.ndarray, size: int = 72) -> Tuple[np.ndarray, np.ndarray]:
    # Resize to square size x size
    T = intensity.shape[0]
    resized_intensity = np.zeros((T, size, size, 1), dtype=np.uint8)  # Single channel
    resized_depth = np.zeros((T, size, size), dtype=np.uint8)
    for i in range(T):
        # Resize intensity and keep only single channel
        intensity_resized = cv2.resize(intensity[i], (size, size), interpolation=cv2.INTER_AREA)
        # intensity_resized is already 2D (size, size) since we took only one channel earlier
        resized_intensity[i, :, :, 0] = intensity_resized
        
        d = depth[i]
        if d.ndim == 3:
            d = d[:, :, 0]
        d_resized = cv2.resize(d, (size, size), interpolation=cv2.INTER_NEAREST)
        # Normalize depth to 0-255 per-frame to fit 8-bit
        d_norm = cv2.normalize(d_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        resized_depth[i] = d_norm

    # Convert to float and standardize per-channel across the clip
    intensity_f = resized_intensity.astype(np.float32) / 255.0
    depth_f = resized_depth.astype(np.float32) / 255.0

    # Standardize intensity channel (single channel)
    intensity_ch = intensity_f[:, :, :, 0]
    intensity_ch = intensity_ch - np.mean(intensity_ch)
    std = np.std(intensity_ch) + 1e-6
    intensity_f[:, :, :, 0] = intensity_ch / std

    # Standardize depth
    depth_f = depth_f - np.mean(depth_f)
    depth_std = np.std(depth_f) + 1e-6
    depth_f = depth_f / depth_std

    return intensity_f, depth_f


def _to_model_inputs(intensity_f: np.ndarray, depth_f: np.ndarray, frame_depth: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    T = intensity_f.shape[0]
    usable_T = (T // frame_depth) * frame_depth
    if usable_T < frame_depth:
        return torch.empty(0), torch.empty(0)

    intensity_f = intensity_f[:usable_T]  # (T,H,W,1)
    depth_f = depth_f[:usable_T]  # (T,H,W)

    # Convert to (T,C,H,W)
    intensity_t = torch.from_numpy(np.transpose(intensity_f, (0, 3, 1, 2))).contiguous()  # (T,1,H,W)
    depth_t = torch.from_numpy(depth_f[:, None, :, :]).contiguous()  # (T,1,H,W)
    return intensity_t, depth_t


def _predict_signal(model: OMNICAN, device: torch.device, intensity_t: torch.Tensor, depth_t: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        intensity_t = intensity_t.to(device)
        depth_t = depth_t.to(device)
        out = model(intensity_t, depth_t)  # (T,1)
        pred = out.squeeze(-1).detach().cpu().numpy()
    return np.asarray(pred, dtype=np.float32)


def _estimate_hr_fft(pred_signal: np.ndarray, fs: int = 30, plot_dir: str = None, clip_name: str = None, signal_type: str = "pred") -> float:
    if pred_signal.size == 0:
        return float('nan')
    
    # Detrend and bandpass similar to calculate_metric_per_video (labels not available)
    sig = pp._detrend(pred_signal, 100)
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
        sig = filtfilt(b, a, np.double(sig))
    except Exception:
        pass
    
    # Calculate HR using the same method as find_HR
    Intensity_freq = np.fft.rfft(sig)
    X_final = np.abs(Intensity_freq)
    
    # Create frequency axis in BPM
    freq = np.fft.rfftfreq(len(sig), 1.0 / fs) * 60.0
    
    # Filter to physiological heart rate range
    mask = (freq >= 50) & (freq <= 150)
    freq_filtered = freq[mask]
    hr_arr = X_final[mask]
    
    # Find peak frequency
    if len(hr_arr) > 0:
        hr = freq_filtered[np.argmax(hr_arr)]
    else:
        print(f"Warning: No valid frequencies found in range 50-150 BPM for {signal_type}")
        hr = 75.0  # Default fallback
    
    # Generate diagnostic plots if plot_dir is provided
    if plot_dir and clip_name:
        try:
            # Create unique filename
            clean_name = str(clip_name).replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            file_prefix = f"{signal_type}_{clean_name}"
            
            # HR frequency spectrum plot
            hr_plot_dir = os.path.join(plot_dir, 'HR_diagram')
            os.makedirs(hr_plot_dir, exist_ok=True)
            
            plt.figure(figsize=(10, 6))
            plt.plot(freq_filtered, hr_arr, 'b-', linewidth=2)
            plt.axvline(hr, color='red', linestyle='--', linewidth=2, 
                       label=f'Peak: {hr:.1f} BPM')
            plt.xlabel('Frequency (BPM)')
            plt.ylabel('FFT Magnitude')
            plt.title(f'HR Frequency Spectrum - {signal_type.upper()} (Video: {clip_name})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(50, 150)
            plt.savefig(os.path.join(hr_plot_dir, f'{file_prefix}_HR_diagram.pdf'), bbox_inches='tight', dpi=300)
            plt.close()
            
            # Time domain waveform plot
            waveform_plot_dir = os.path.join(plot_dir, 'waveform')
            os.makedirs(waveform_plot_dir, exist_ok=True)
            
            time_axis = np.arange(len(sig)) / fs
            plt.figure(figsize=(12, 6))
            plt.plot(time_axis, sig, 'b-', linewidth=1, alpha=0.8)
            plt.xlabel('Time (seconds)')
            plt.ylabel('PPG Amplitude')
            plt.title(f'PPG Waveform - {signal_type.upper()} (Estimated HR: {hr:.1f} BPM) - Video: {clip_name}')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(waveform_plot_dir, f'{file_prefix}_waveform.pdf'), bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"  Saved {signal_type} plots: HR={hr:.1f} BPM")
            
        except Exception as e:
            print(f"Warning: Could not save diagnostic plots for {signal_type}: {e}")
    
    return float(hr)


def _estimate_hr_find_hr(signal: np.ndarray, fs: int = 30, plot_dir: str = None, clip_name: str = None, signal_type: str = "gt") -> float:
    """Calculate HR using the find_HR method from post_process.py for ground truth."""
    if signal.size == 0:
        return float('nan')
    
    # Use the find_HR method directly
    try:
        # Create a mock config object for find_HR
        class MockConfig:
            class LOG:
                PATH = plot_dir if plot_dir else os.getcwd()
            class TEST:
                class DATA:
                    EXP_DATA_NAME = "inference"
        
        config = MockConfig()
        hr = pp.find_HR(signal, config, signal_type, fs, video_index=clip_name)
        
        print(f"  Calculated {signal_type} HR using find_HR: {hr:.1f} BPM")
        return float(hr)
        
    except Exception as e:
        print(f"Warning: Could not use find_HR for {signal_type}: {e}")
        # Fallback to our FFT method
        return _estimate_hr_fft(signal, fs, plot_dir, clip_name, signal_type)


def _resample_gt_ppg(gt_ppg: np.ndarray, target_length: int, gt_fps: int = 60, video_fps: int = 20) -> np.ndarray:
    """Resample ground truth PPG to match video frame length accounting for different sampling rates."""
    if gt_ppg.size == 0:
        return np.empty((0,))
    
    # Calculate the correct target length based on sampling rate ratio
    # If GT is at 60fps and video is at 20fps, we need to downsample by factor of 3
    # But we want to match the video frame count, not the time duration
    sampling_ratio = gt_fps / video_fps
    corrected_target_length = int(target_length * sampling_ratio)
    
    print(f"  Resampling GT: {len(gt_ppg)} samples -> {corrected_target_length} samples (ratio: {sampling_ratio:.2f})")
    
    # Resample to the corrected length
    resampled = BaseLoader.resample_ppg(gt_ppg, corrected_target_length)
    
    # Then downsample to match video frame count
    if len(resampled) > target_length:
        # Downsample by taking every nth sample
        step = len(resampled) // target_length
        final_resampled = resampled[::step][:target_length]
    else:
        final_resampled = resampled
    
    print(f"  Final GT length: {len(final_resampled)} samples")
    return final_resampled


def _load_model(checkpoint_path: str, frame_depth: int = 10, img_size: int = 72, device: torch.device = torch.device('cpu')) -> OMNICAN:
    # Instantiate with 1-channel intensity and 1-channel depth
    model = OMNICAN(in_channels=1, depth_channels=1, frame_depth=frame_depth, img_size=img_size).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get('state_dict', state)
    # Strip DataParallel prefixes if present
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state[k[len('module.'):]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    return model


def run_inference(root_dir: str, checkpoint_path: str, fs: int = 30, frame_depth: int = 10, img_size: int = 72, device_str: str = 'cuda:0', display_frames: bool = True, save_plots: bool = True, plot_dir: str = None, gt_fps: int = 60, video_fps: int = 20):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    model = _load_model(checkpoint_path, frame_depth=frame_depth, img_size=img_size, device=device)

    clip_dirs = _list_clip_dirs(root_dir, max_clips=10)
    print(f"Found {len(clip_dirs)} clips")
    results = []
    
    # Set up plot directory
    # save plots to current_date_time_inference_plots
    if save_plots and plot_dir is None:
        plot_dir = os.path.join(os.getcwd(), f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_inference_plots')
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Plots will be saved to: {plot_dir}")
    
    print("=" * 80)
    print("OMNI-CAN Inference with Ground Truth Comparison")
    print("=" * 80)
    
    for clip in clip_dirs:
        clip_name = os.path.basename(clip)
        print(f"\nProcessing: {clip_name}")
        
        # Load intensity and depth frames
        intensity, depth = _read_rgbd_frames(clip, display_frames=display_frames)
        if intensity.size == 0:
            print(f"  Skipping: no frames found")
            continue
            
        # Load ground truth PPG
        gt_ppg = _read_gt_ppg(clip)
        if gt_ppg.size == 0:
            print(f"  Warning: no ground truth PPG found")
            gt_ppg = None
        else:
            # Resample GT PPG to match video frame length accounting for different sampling rates
            orig_len = len(gt_ppg)
            gt_ppg = _resample_gt_ppg(gt_ppg, len(intensity), gt_fps=gt_fps, video_fps=video_fps)
            print(f"  Loaded GT PPG: {len(gt_ppg)} samples, original length: {orig_len}")
            print(f"  GT PPG now effectively at {video_fps} Hz (was {gt_fps} Hz)")
        
        # Preprocess frames
        intensity_f, depth_f = _resize_and_standardize(intensity, depth, size=img_size)
        intensity_t, depth_t = _to_model_inputs(intensity_f, depth_f, frame_depth=frame_depth)
        if intensity_t.numel() == 0:
            print(f"  Skipping: too few frames for frame_depth={frame_depth}")
            continue
            
        # Run model inference
        pred_signal = _predict_signal(model, device, intensity_t, depth_t)
        pred_hr = _estimate_hr_fft(pred_signal, fs=fs, plot_dir=plot_dir if save_plots else None, 
                                  clip_name=clip_name, signal_type="pred")
        
        # Compute ground truth HR if available
        if gt_ppg is not None and gt_ppg.size > 0:
            # Align GT PPG to predicted signal length
            if len(gt_ppg) != len(pred_signal):
                gt_ppg = _resample_gt_ppg(gt_ppg, len(pred_signal), gt_fps=gt_fps, video_fps=video_fps)
            # Use find_HR method for ground truth with correct sampling rate
            # The resampled GT PPG is now at video_fps, not the original fs
            gt_hr = _estimate_hr_find_hr(gt_ppg, fs=video_fps, plot_dir=plot_dir if save_plots else None, 
                                       clip_name=clip_name, signal_type="gt")
            
            # Calculate error metrics
            hr_error = abs(pred_hr - gt_hr)
            hr_error_pct = (hr_error / gt_hr) * 100 if gt_hr > 0 else float('inf')
            
            print(f"  Predicted HR: {pred_hr:.2f} BPM")
            print(f"  Ground Truth HR: {gt_hr:.2f} BPM")
            print(f"  Error: {hr_error:.2f} BPM ({hr_error_pct:.1f}%)")
            
            results.append((clip_name, pred_hr, gt_hr, hr_error, hr_error_pct))
        else:
            print(f"  Predicted HR: {pred_hr:.2f} BPM")
            print(f"  Ground Truth: Not available")
            results.append((clip_name, pred_hr, None, None, None))
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    valid_results = [r for r in results if r[2] is not None]  # Results with GT
    if valid_results:
        pred_hrs = [r[1] for r in valid_results]
        gt_hrs = [r[2] for r in valid_results]
        errors = [r[3] for r in valid_results]
        error_pcts = [r[4] for r in valid_results]
        
        print(f"Clips with ground truth: {len(valid_results)}")
        print(f"Mean Predicted HR: {np.mean(pred_hrs):.2f} ± {np.std(pred_hrs):.2f} BPM")
        print(f"Mean Ground Truth HR: {np.mean(gt_hrs):.2f} ± {np.std(gt_hrs):.2f} BPM")
        print(f"Mean Absolute Error: {np.mean(errors):.2f} ± {np.std(errors):.2f} BPM")
        print(f"Mean Error Percentage: {np.mean(error_pcts):.2f} ± {np.std(error_pcts):.2f}%")
        
        # Calculate correlation
        correlation = np.corrcoef(pred_hrs, gt_hrs)[0, 1]
        print(f"Pearson Correlation: {correlation:.3f}")
    
    no_gt_results = [r for r in results if r[2] is None]  # Results without GT
    if no_gt_results:
        print(f"\nClips without ground truth: {len(no_gt_results)}")
        for clip_name, pred_hr, _, _, _ in no_gt_results:
            print(f"  {clip_name}: {pred_hr:.2f} BPM")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r'D:\Github repos\data-collector\Data', help='Root directory containing clip subfolders')
    parser.add_argument('--checkpoint', type=str, default=os.path.join('neural_methods', 'checkpoints', 'OMNICAN_GD_Epoch19.pth'))
    parser.add_argument('--fs', type=int, default=30)
    parser.add_argument('--frame_depth', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=72)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--display_frames', action='store_true', default=True, help='Display intensity frames during loading')
    parser.add_argument('--no_display', action='store_true', help='Disable frame display')
    parser.add_argument('--save_plots', action='store_true', default=True, help='Save HR diagnostic plots')
    parser.add_argument('--no_plots', action='store_true', help='Disable plot saving')
    parser.add_argument('--plot_dir', type=str, default=None, help='Directory to save plots (default: ./inference_plots)')
    parser.add_argument('--gt_fps', type=int, default=60, help='Ground truth PPG sampling rate (default: 60 Hz)')
    parser.add_argument('--video_fps', type=int, default=20, help='Video frame rate (default: 20 Hz)')
    args = parser.parse_args()

    # Handle display flag
    display_frames = args.display_frames and not args.no_display
    save_plots = args.save_plots and not args.no_plots

    run_inference(
        root_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        fs=args.fs,
        frame_depth=args.frame_depth,
        img_size=args.img_size,
        device_str=args.device,
        display_frames=display_frames,
        save_plots=save_plots,
        plot_dir=args.plot_dir,
        gt_fps=args.gt_fps,
        video_fps=args.video_fps,
    )


if __name__ == '__main__':
    main()


