from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import gradio as gr
from orpheus import generate_long_form_speech, load_model, cleanup_files
import uvicorn

app = FastAPI(
    title="OrpheusTTS API",
    description="API for OrpheusTTS long-form speech generation",
    version="1.0.0"
)

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
    audio_file: str
    stats: str

@app.on_event("startup")
async def startup_event():
    """Load the TTS model when the API starts"""
    load_model()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up generated files when the API shuts down"""
    cleanup_files()

@app.post("/generate/", response_model=TTSResponse)
async def generate_speech(request: TTSRequest):
    """
    Generate long-form speech from text using OrpheusTTS.
    
    Returns:
        - audio_file: Path to the generated WAV file
        - stats: Generation statistics
    """
    try:
        # Create a mock progress object since we're not using the Gradio UI
        mock_progress = gr.Progress()
        
        audio_file, stats = generate_long_form_speech(
            long_text=request.text,
            voice=request.voice,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            batch_size=request.batch_size,
            max_tokens=request.max_tokens,
            progress=mock_progress
        )
        
        return TTSResponse(audio_file=audio_file, stats=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
