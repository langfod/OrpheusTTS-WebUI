

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from typing import Optional
import logging
import tempfile
import wave
import os
from orpheus_onnx_model import OrpheusONNXModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OrpheusTTS ONNX API",
    description="API for OrpheusTTS ONNX speech generation using ONNX runtime",
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

class TTSResponse(BaseModel):
    message: str
    audio_length_seconds: Optional[float] = None
    model_info: dict

@app.on_event("startup")
async def startup_event():
    """Load the ONNX TTS model when the API starts"""
    global model
    try:
        logger.info("Starting ONNX model loading...")
        model = OrpheusONNXModel(model_name="Prince-1/OrpheusTTS-ONNX")
        logger.info("ONNX model loaded successfully")
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

if __name__ == "__main__":
    uvicorn.run("orpheus_onnx_api:app", host="0.0.0.0", port=8000, reload=True)
