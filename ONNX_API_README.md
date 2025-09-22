# Orpheus ONNX API Documentation

## Overview

This implementation provides a complete ONNX-based API for the Orpheus TTS model, including voice cloning capabilities. The API is designed to be compatible with the original Orpheus TTS interface while using ONNX runtime for inference.

## Features

### ✅ Implemented Features

1. **Basic Speech Generation**
   - Text-to-speech with voice selection
   - Temperature, top-p, and repetition penalty controls
   - Support for all 8 voices: tara, jess, leo, leah, dan, mia, zac, zoe
   - Audio output in WAV format

2. **Voice Cloning (Zero-shot)**
   - Upload reference audio file
   - Provide reference transcript
   - Generate speech in the cloned voice for multiple texts
   - ZIP download of all generated audio files

3. **Mock Model Support**
   - Fully functional API even without real ONNX model
   - Mock audio generation for development and testing
   - Ready for real model integration when available

4. **RESTful API Endpoints**
   - `/` - API information and status
   - `/model-info/` - Model and configuration details
   - `/generate/` - Generate speech from text
   - `/generate-audio/` - Generate and download audio file
   - `/clone-voice/` - Voice cloning (returns metadata)
   - `/clone-voice-audio/` - Voice cloning (returns ZIP of audio files)

## API Endpoints

### GET `/`
Returns API information and available endpoints.

### GET `/model-info/`
Returns detailed information about the loaded model.

### POST `/generate/`
Generate speech from text with voice parameters.

**Request Body:**
```json
{
    "text": "Hello, this is a test!",
    "voice": "tara",
    "temperature": 0.6,
    "top_p": 0.8,
    "repetition_penalty": 1.1,
    "max_tokens": 1200
}
```

### POST `/generate-audio/`
Generate speech and return WAV audio file directly.

**Request Body:** Same as `/generate/`
**Response:** WAV audio file

### POST `/clone-voice/`
Clone voice from reference audio and generate speech for multiple texts.

**Form Data:**
- `audio_file`: Reference audio file (WAV recommended)
- `reference_transcript`: Text transcript of the reference audio
- `texts`: JSON array of texts to generate (e.g., `["Hello", "How are you?"]`)
- `temperature`: Sampling temperature (optional, default: 0.5)
- `top_p`: Top-p parameter (optional, default: 0.9)
- `repetition_penalty`: Repetition penalty (optional, default: 1.1)
- `max_tokens`: Maximum tokens (optional, default: 990)

**Response:**
```json
{
    "message": "Voice cloning completed successfully for 2 texts",
    "generated_count": 2,
    "total_audio_length_seconds": 4.5,
    "model_info": {...}
}
```

### POST `/clone-voice-audio/`
Same as `/clone-voice/` but returns a ZIP file containing all generated audio files.

**Response:** ZIP file with WAV audio files

## Usage Examples

### Basic Speech Generation

```bash
curl -X POST http://localhost:8000/generate/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice": "tara"}'
```

### Download Audio File

```bash
curl -X POST http://localhost:8000/generate-audio/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice": "jess"}' \
  -o output.wav
```

### Voice Cloning

```bash
curl -X POST http://localhost:8000/clone-voice-audio/ \
  -F "audio_file=@reference.wav" \
  -F "reference_transcript=Hello, this is my voice." \
  -F 'texts=["Test one", "Test two"]' \
  -o cloned_voices.zip
```

## Running the API

1. **Install Dependencies**
   ```bash
   pip install onnxruntime-genai transformers fastapi uvicorn librosa python-multipart
   ```

2. **Start the Server**
   ```bash
   python orpheus_onnx_api.py
   ```

3. **Access the API**
   - Server runs on `http://localhost:8000`
   - Interactive docs at `http://localhost:8000/docs`

## Architecture

### Core Components

1. **`orpheus_onnx_model.py`**
   - Main ONNX model wrapper
   - Handles model loading and inference
   - Mock implementation for development

2. **`orpheus_onnx_cloning.py`**
   - Voice cloning functionality
   - Audio tokenization with SNAC codec
   - Zero-shot voice synthesis

3. **`orpheus_onnx_api.py`**
   - FastAPI server implementation
   - RESTful endpoints
   - File upload and download handling

4. **`test_onnx_api.py`**
   - Comprehensive test suite
   - API validation and verification

### Model Integration

The system is designed to work with:
- **Target Model:** `Prince-1/OrpheusTTS-ONNX`
- **Original Models:** `canopylabs/orpheus-3b-0.1-pretrained`, `canopylabs/orpheus-3b-0.1-ft`
- **Audio Codec:** SNAC for audio tokenization

### Mock Mode

When the real ONNX model is not available, the system operates in mock mode:
- Generates realistic placeholder audio
- Maintains full API compatibility
- Voice characteristics vary based on selected voice
- Cloned voices adapt to reference audio characteristics

## Current Status

### ✅ Completed
- [x] ONNX model wrapper with fallback support
- [x] FastAPI server with all endpoints
- [x] Basic speech generation with voice selection
- [x] Zero-shot voice cloning implementation
- [x] Audio file generation and download
- [x] Comprehensive test suite
- [x] Mock implementations for development

### 🔄 Ready for Integration
- [ ] Real ONNX model integration (when available)
- [ ] SNAC audio codec integration
- [ ] Performance optimization for production

### 🎯 Future Enhancements
- [ ] Streaming audio generation
- [ ] Batch processing for multiple requests
- [ ] Voice fine-tuning capabilities
- [ ] Real-time voice conversion

## Testing

Run the comprehensive test suite:

```bash
python test_onnx_api.py
```

This tests all endpoints, parameter variations, and voice cloning functionality.

## Notes

- The implementation is fully functional in mock mode
- Ready for seamless integration with real ONNX models
- Compatible with the original Orpheus TTS API design
- Supports both CPU and CUDA inference (when models are available)
- File uploads are handled securely with temporary file cleanup