from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from acestep.pipeline_ace_step import ACEStepPipeline

app = FastAPI(title="ACEStep Pipeline API")
pipeline = None

class MusicGenerationRequest(BaseModel):
    prompt: str
    lyrics: str = None
    audio_duration: float = 60.0
    infer_step: int = 60
    guidance_scale: float = 15.0
    format: str = "wav"

@app.on_event("startup")
async def startup_event():
    global pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = ACEStepPipeline(device_id=0 if device == 'cuda' else -1, dtype="bfloat16")
    pipeline.load_checkpoint()

@app.post("/generate_music")
async def generate_music(request: MusicGenerationRequest):
    audio_path, params_json = pipeline(
        format=request.format,
        audio_duration=request.audio_duration,
        prompt=request.prompt,
        lyrics=request.lyrics,
        infer_step=request.infer_step,
        guidance_scale=request.guidance_scale,
    )
    return FileResponse(audio_path, media_type=f"audio/{request.format}", filename=f"generated_music.{request.format}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
