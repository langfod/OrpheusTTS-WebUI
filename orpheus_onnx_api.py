

from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import tempfile
import wave
import os
from orpheus_onnx_model import OrpheusONNXModel
from orpheus_onnx_cloning import add_cloning_to_model
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OrpheusTTS ONNX API",
    description="API for OrpheusTTS ONNX speech generation and voice cloning using ONNX runtime",
    version="1.0.0"
)

# Global model instance
model: Optional[OrpheusONNXModel] = None

# Define the request model with the same defaults as the Gradio UI
class TTSRequest(BaseModel):
    text: str = Field(..., description="The text to convert to speech")
    voice: str = Field(
        default="tara",
        description="Voice to use for speech generation",
        enum=["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"]
    )
    temperature: float = Field(
        default=0.6,
        description="Temperature for speech generation",
        ge=0.1,
        le=2.0
    )
    top_p: float = Field(
        default=0.8,
        description="Top P value for speech generation",
        ge=0.1,
        le=1.0
    )
    repetition_penalty: float = Field(
        default=1.1,
        description="Repetition penalty for speech generation",
        ge=1.0,
        le=2.0
    )
    batch_size: int = Field(
        default=4,
        description="Number of chunks to process in parallel",
        ge=1,
        le=10
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for speech generation",
        ge=128,
        le=16384
    )

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello! This is a test of the long-form speech generation.",
                "voice": "tara",
                "temperature": 0.6,
                "top_p": 0.8,
                "repetition_penalty": 1.1,
                "batch_size": 4,
                "max_tokens": 4096
            }
        }

class CloningRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to generate in the cloned voice")
    reference_transcript: str = Field(..., description="Transcript of the reference audio")
    temperature: float = Field(
        default=0.5,
        description="Temperature for speech generation",
        ge=0.1,
        le=2.0
    )
    top_p: float = Field(
        default=0.9,
        description="Top P value for speech generation",
        ge=0.1,
        le=1.0
    )
    repetition_penalty: float = Field(
        default=1.1,
        description="Repetition penalty for speech generation",
        ge=1.0,
        le=2.0
    )
    max_tokens: int = Field(
        default=990,
        description="Maximum tokens for speech generation",
        ge=128,
        le=2048
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Hello, this is a test of voice cloning!", "How are you doing today?"],
                "reference_transcript": "Hi there, my name is Sarah and I'm excited to try this technology.",
                "temperature": 0.5,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "max_tokens": 990
            }
        }

class TTSResponse(BaseModel):
    message: str
    audio_length_seconds: Optional[float] = None
    model_info: dict

class CloningResponse(BaseModel):
    message: str
    generated_count: int
    total_audio_length_seconds: float
    model_info: dict

@app.on_event("startup")
async def startup_event():
    """Load the ONNX TTS model when the API starts"""
    global model
    try:
        logger.info("Starting ONNX model loading...")
        model = OrpheusONNXModel(model_name="Prince-1/OrpheusTTS-ONNX")
        # Add voice cloning capability
        model = add_cloning_to_model(model)
        logger.info("ONNX model loaded successfully with cloning support")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        # Don't fail startup, but model will be None

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up when the API shuts down"""
    global model
    model = None
    logger.info("ONNX API shutdown complete")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "OrpheusTTS ONNX API",
        "version": "1.0.0",
        "status": "ready" if model is not None else "model_not_loaded",
        "endpoints": {
            "/generate/": "POST - Generate speech from text",
            "/generate-audio/": "POST - Generate speech and return audio file",
            "/clone-voice/": "POST - Clone voice from reference audio",
            "/clone-voice-audio/": "POST - Clone voice and return audio files",
            "/model-info/": "GET - Get model information"
        }
    }

@app.get("/model-info/")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return model.get_model_info()

@app.post("/generate/", response_model=TTSResponse)
async def generate_speech(request: TTSRequest):
    """
    Generate speech from text using OrpheusTTS ONNX model.
    
    Returns:
        - message: Status message
        - audio_length_seconds: Length of generated audio
        - model_info: Information about the model used
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Generating speech for text: {request.text[:50]}...")
        
        # Generate speech using ONNX model
        audio_data = model.generate_speech(
            prompt=request.text,
            voice=request.voice,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            max_tokens=request.max_tokens
        )
        
        # Calculate audio length (rough estimate)
        audio_length = len(audio_data) / (model.sample_rate * 2)  # 16-bit audio
        
        return TTSResponse(
            message=f"Speech generated successfully for {len(request.text)} characters",
            audio_length_seconds=audio_length,
            model_info=model.get_model_info()
        )
        
    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-audio/")
async def generate_speech_audio(request: TTSRequest):
    """
    Generate speech and return the audio file directly.
    
    Returns:
        WAV audio file as response
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Generating audio for text: {request.text[:50]}...")
        
        # Generate speech using ONNX model
        audio_data = model.generate_speech(
            prompt=request.text,
            voice=request.voice,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            max_tokens=request.max_tokens
        )
        
        # Create a WAV file in memory
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(model.sample_rate)
                wav_file.writeframes(audio_data)
            
            # Read the file content
            with open(tmp_file.name, 'rb') as f:
                wav_content = f.read()
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
        
        return Response(
            content=wav_content,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=orpheus_output.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clone-voice/", response_model=CloningResponse)
async def clone_voice(
    audio_file: UploadFile = File(..., description="Reference audio file (WAV format recommended)"),
    reference_transcript: str = Form(..., description="Transcript of the reference audio"),
    texts: str = Form(..., description="JSON array of texts to generate (e.g., [\"Hello\", \"How are you?\"])"),
    temperature: float = Form(0.5, description="Sampling temperature"),
    top_p: float = Form(0.9, description="Top-p sampling parameter"),
    repetition_penalty: float = Form(1.1, description="Repetition penalty"),
    max_tokens: int = Form(990, description="Maximum tokens to generate")
):
    """
    Clone a voice from reference audio and generate speech for multiple texts.
    
    Upload a reference audio file and provide its transcript to clone the voice,
    then generate speech for the provided texts in that voice.
    """
    if model is None or not hasattr(model, 'cloning'):
        raise HTTPException(status_code=503, detail="Model not loaded or cloning not available")
    
    try:
        # Parse texts JSON
        import json
        try:
            text_list = json.loads(texts)
            if not isinstance(text_list, list):
                raise ValueError("texts must be a JSON array")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in texts parameter: {e}")
        
        logger.info(f"Voice cloning request: {len(text_list)} texts")
        
        # Save uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_audio_path = tmp_file.name
        
        try:
            # Perform voice cloning
            audio_results = model.cloning.clone_voice(
                reference_audio_path=tmp_audio_path,
                reference_transcript=reference_transcript,
                target_texts=text_list,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens
            )
            
            # Calculate total audio length
            total_length = 0.0
            for audio_data in audio_results:
                audio_length = len(audio_data) / (model.sample_rate * 2)
                total_length += audio_length
            
            return CloningResponse(
                message=f"Voice cloning completed successfully for {len(text_list)} texts",
                generated_count=len(audio_results),
                total_audio_length_seconds=total_length,
                model_info=model.get_model_info()
            )
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_audio_path)
        
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clone-voice-audio/")
async def clone_voice_audio(
    audio_file: UploadFile = File(..., description="Reference audio file (WAV format recommended)"),
    reference_transcript: str = Form(..., description="Transcript of the reference audio"),
    texts: str = Form(..., description="JSON array of texts to generate"),
    temperature: float = Form(0.5, description="Sampling temperature"),
    top_p: float = Form(0.9, description="Top-p sampling parameter"),
    repetition_penalty: float = Form(1.1, description="Repetition penalty"),
    max_tokens: int = Form(990, description="Maximum tokens to generate")
):
    """
    Clone a voice and return the generated audio files as a ZIP archive.
    
    This endpoint performs voice cloning and returns all generated audio files
    packaged in a ZIP file for easy download.
    """
    if model is None or not hasattr(model, 'cloning'):
        raise HTTPException(status_code=503, detail="Model not loaded or cloning not available")
    
    try:
        # Parse texts JSON
        import json
        try:
            text_list = json.loads(texts)
            if not isinstance(text_list, list):
                raise ValueError("texts must be a JSON array")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in texts parameter: {e}")
        
        logger.info(f"Voice cloning audio request: {len(text_list)} texts")
        
        # Save uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_audio_path = tmp_file.name
        
        try:
            # Perform voice cloning
            audio_results = model.cloning.clone_voice(
                reference_audio_path=tmp_audio_path,
                reference_transcript=reference_transcript,
                target_texts=text_list,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens
            )
            
            # Create ZIP file with all generated audio
            import zipfile
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as zip_file:
                with zipfile.ZipFile(zip_file.name, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for i, (audio_data, text) in enumerate(zip(audio_results, text_list)):
                        # Create WAV file in memory
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
                            with wave.open(wav_file.name, 'wb') as wav:
                                wav.setnchannels(1)  # Mono
                                wav.setsampwidth(2)  # 16-bit
                                wav.setframerate(model.sample_rate)
                                wav.writeframes(audio_data)
                            
                            # Add to ZIP with descriptive name
                            safe_text = "".join(c for c in text[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
                            filename = f"cloned_voice_{i+1:02d}_{safe_text}.wav"
                            zf.write(wav_file.name, filename)
                            os.unlink(wav_file.name)
                
                # Read ZIP content
                with open(zip_file.name, 'rb') as f:
                    zip_content = f.read()
                
                os.unlink(zip_file.name)
            
            return Response(
                content=zip_content,
                media_type="application/zip",
                headers={
                    "Content-Disposition": "attachment; filename=cloned_voice_audio.zip"
                }
            )
            
        finally:
            # Clean up temporary audio file
            os.unlink(tmp_audio_path)
        
    except Exception as e:
        logger.error(f"Voice cloning audio generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("orpheus_onnx_api:app", host="0.0.0.0", port=8000, reload=True)
