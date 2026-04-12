# ml-asr

## Overview

This project implements a Speech Enhancement (Denoising) system using a Band-Split Recurrent Neural Network (BSRNN). Built with PyTorch, the repository includes a complete pipeline from training on noisy audio to an interactive Flask-based Minimum Viable Product (MVP) web interface. The system allows users to record their voice, apply synthetic environmental noise (Babble, Room Impulse Response, or White Noise), and process it through the neural network to recover clean speech.

## Project Goals

    Effective Noise Suppression: Implement a state-of-the-art BSRNN architecture to separate clean speech from background noise in the frequency domain.

    Robustness to Diverse Environments: Train the model to generalize across various acoustic conditions, including multi-talker babble, reverberation (RIR), and white noise.

    ASR Improvement: Measurably improve speech recognition accuracy (Word Error Rate) on degraded audio using modern ASR models like Whisper.

    Accessible Interface: Provide an easy-to-use web UI to demonstrate real-time audio capture and model inference capabilities.

## Data

The model is trained using a combination of clean speech and synthetic noise overlays:

    Clean Speech: Sourced from the Hugging Face dataset fsicoli/common_voice_22_0 (Russian language split, utilizing ru_train_0_19.tar).

    Noise Assets: * Babble Noise: Multi-talker background noise from the Lab41-SRI-VOiCES corpus.

        Reverberation (RIR): Room Impulse Responses from the Lab41-SRI-VOiCES corpus to simulate echoes and spatial acoustics.

        White Noise: Synthetically generated normal distribution noise.

## Methodology

The training pipeline utilizes on-the-fly data augmentation. During training, clean speech segments are randomly mixed with one of the noise profiles at varying Signal-to-Noise Ratios (SNR) ranging from -5 dB to 15 dB.

To optimize the model, a custom Combined Loss Function is used:

    SI-SDR (Scale-Invariant Signal-to-Distortion Ratio): Focuses on the time-domain waveform alignment and energy matching.

    Multi-Resolution STFT Loss: Computes Spectral Convergence and Log-Magnitude loss across multiple FFT sizes (512, 1024, 2048) to ensure high fidelity in the frequency domain.

## Model

The core architecture is defined in model.py and relies on the Band-Split RNN (BSRNN) design. It operates in the complex Short-Time Fourier Transform (STFT) domain. The architecture consists of four main components:

    BandSplitModule: Divides the input spectrogram into multiple sub-bands with specific target bandwidths, normalizing and projecting each band independently.

    TemporalModel: A sequential LSTM that models temporal dependencies within each frequency band across time steps.

    BandModel: A bi-directional LSTM that models the relationships across different frequency bands at a single time step.

    MaskEstimator: Generates a complex mask (Real and Imaginary parts) which is applied to the noisy input spectrogram to filter out the noise, followed by an Inverse STFT to reconstruct the time-domain waveform.

## Installation

Ensure you have Python 3.8+ installed. You can install the required dependencies using pip:
Bash

Install PyTorch and Torchaudio (version 2.8.0 recommended per training script)
pip install torch==2.8.0 torchaudio==2.8.0

Install remaining dependencies
pip install Flask soundfile numpy pandas librosa openai-whisper jiwer tqdm ipython

## How to Run

To start the app, install all important librarioes by running:

pip install -r requirements.txt

then, ensure the trained model weights (models/bsrnn_weights_final.pth) are placed in the correct directory. Then, launch the Flask app:

python app.py

Open your browser and navigate to http://127.0.0.1:5000.

## Example

Once the Flask web application is running:

    Open the UI in your browser.

    Select a Noise Type from the dropdown menu (e.g., "Babble", "RIR", "White").

    Click "Начать запись" (Start Recording) and speak into your microphone.

    Click "Остановить и обработать" (Stop and Process).

    The backend will automatically mix your voice with the selected noise, run the BSRNN inference, and return three audio players:

        Clean: Your original recording.

        Noisy: Your recording with the synthetic noise applied.

        Denoised: The neural network's cleaned output.

## Evaluation

The evaluation pipeline (in BRSNN.py) utilizes OpenAI's Whisper (large-v3) to transcribe the audio. The effectiveness of the model is measured using:

    Word Error Rate (WER): Compares the Whisper transcriptions of the clean, noisy, and denoised audio to the ground-truth text.

    SI-SDR Improvement (dB): Measures the absolute decibel gain in signal clarity between the noisy input and the denoised output.

## Limitations

    Sample Rate Restriction: The model strictly operates at a 16 kHz sample rate. Files with different rates are automatically resampled, which may lead to loss of high-frequency data.

    Compute Heavy: Inference relies on complex STFT conversions and multiple LSTM layers, which can introduce latency if run on CPU rather than a CUDA-enabled GPU.

    Browser Formatting: WebM audio recorded directly from standard browsers requires specific handling. Currently, the backend reads it via soundfile, which may occasionally require an intermediate ffmpeg conversion depending on browser implementations.

## Future Improvements

    Real-time Streaming Inference: Porting the block-based STFT processing to handle continuous buffer streams rather than file-by-file processing.

    Dataset Expansion: Training on a wider variety of languages and environmental noise sets (e.g., MUSAN dataset, UrbanSound8K) to improve zero-shot generalization.

    Quantization: Applying INT8 quantization or ONNX runtime optimization to speed up CPU inference for edge deployment.

## License

This project is licensed under the MIT License.
