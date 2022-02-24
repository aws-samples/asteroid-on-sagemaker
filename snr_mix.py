import pathlib
import numpy as np
# import soundfile as sf

# import librosa


EPS = np.finfo(float).eps
np.random.seed(0)

def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)

def normalize(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio

def snr_mixer(clean, noise, snr, target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various SNR levels'''
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean)-len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise)-len(clean)))

    # Normalizing to -25 dB FS
    clean = clean/(max(abs(clean))+EPS)
    clean = normalize(clean, target_level)
    rmsclean = (clean**2).mean()**0.5

    noise = noise/(max(abs(noise))+EPS)
    noise = normalize(noise, target_level)
    rmsnoise = (noise**2).mean()**0.5

    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10**(snr/20)) / (rmsnoise+EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    
    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue. 
    noisy_rms_level = np.random.randint(-35, -15)
    rmsnoisy = (noisyspeech**2).mean()**0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy+EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech))/(clipping_threshold-EPS)
        noisyspeech = noisyspeech/noisyspeech_maxamplevel
        clean = clean/noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel/noisyspeech_maxamplevel
        noisy_rms_level = int(20*np.log10(scalarnoisy/noisyspeech_maxamplevel*(rmsnoisy+EPS)))

    return clean, noisenewlevel, noisyspeech, noisy_rms_level


cleanpath = pathlib.Path('clean').glob('*wav')
noisepath = pathlib.Path('noise').glob('*wav')

for clean, noise in zip(cleanpath, noisepath):
    clean_wav,_ = librosa.load(clean, sr =16000)
    noise_wav,_ = librosa.load(noise, sr = 16000) 
    clean_wav[::2]
    print(clean, noise)