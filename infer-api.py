import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import torch
from acestep.pipeline_ace_step import ACEStepPipeline
import os

app = FastAPI(title="ACEStep Pipeline API", docs_url="/docs")

# Монтируем директорию "outputs" для статических файлов
app.mount("/static", StaticFiles(directory="outputs"), name="static")

# Определение пресетов жанров
GENRE_PRESETS = {
    "Modern Pop": "pop, synth, drums, guitar, 120 bpm, upbeat, catchy, vibrant",
    "Rock": "rock, electric guitar, drums, bass, 130 bpm, energetic, rebellious, gritty",
    "Hip Hop": "hip hop, 808 bass, hi-hats, synth, 90 bpm, bold, urban, intense",
    "Country": "country, acoustic guitar, steel guitar, fiddle, 100 bpm, heartfelt, rustic, warm",
    "EDM": "edm, synth, bass, kick drum, 128 bpm, euphoric, pulsating, energetic, instrumental",
    "Reggae": "reggae, guitar, bass, drums, 80 bpm, chill, soulful, positive",
    "Classical": "classical, orchestral, strings, piano, 60 bpm, elegant, emotive, timeless, instrumental",
    "Jazz": "jazz, saxophone, piano, double bass, 90 bpm, smooth, improvisational, soulful",
    "Metal": "metal, electric guitar, double kick drum, bass, 160 bpm, aggressive, intense, heavy",
    "R&B": "r&b, synth, bass, drums, 85 bpm, sultry, groovy, romantic"
}

pipeline = None

# Модель запроса с поддержкой выбора жанра и инструментальной музыки
class MusicGenerationRequest(BaseModel):
    prompt: Optional[str] = Field(None, description="Промпт для генерации музыки. Игнорируется, если указан genre_preset.")
    genre_preset: Optional[str] = Field(None, description=f"Пресет жанра. Доступные пресеты: {', '.join(GENRE_PRESETS.keys())}")
    lyrics: Optional[str] = Field(None, description="Текст песни. Для инструментальной музыки можно использовать '[instrumental]'.")
    instrumental_only: bool = Field(True, description="Если True, генерирует инструментальную музыку, устанавливая lyrics в '[instrumental]' и игнорируя указанные lyrics.")
    audio_duration: float = 60.0
    infer_step: int = 155
    guidance_scale: float = 25.0
    guidance_interval: float = 0.75
    guidance_interval_decay: float = 1.0
    min_guidance_scale: float = 3.0
    use_erg_tag: bool = True
    use_erg_lyric: bool = False
    use_erg_diffusion: bool = True
    cfg_type: str = "apg"
    scheduler_type: str = "euler"
    omega_scale: float = 10.0
    format: str = "wav"

@app.on_event("startup")
async def startup_event():
    global pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = ACEStepPipeline(device_id=0 if device == 'cuda' else -1, dtype="bfloat16")
    pipeline.load_checkpoint()

@app.post("/generate_music")
async def generate_music(request: MusicGenerationRequest):
    # Проверка взаимоисключающих параметров
    if request.genre_preset and request.prompt:
        raise HTTPException(status_code=422, detail="Нельзя указывать одновременно prompt и genre_preset")
    if not request.genre_preset and not request.prompt:
        raise HTTPException(status_code=422, detail="Необходимо указать либо prompt, либо genre_preset")
    
    # Определение prompt на основе входных данных
    if request.genre_preset:
        if request.genre_preset not in GENRE_PRESETS:
            raise HTTPException(status_code=422, detail=f"Недопустимый genre_preset. Доступные пресеты: {', '.join(GENRE_PRESETS.keys())}")
        prompt = GENRE_PRESETS[request.genre_preset]
    else:
        prompt = request.prompt
    
    # Установка lyrics в "[instrumental]" для инструментальной музыки
    if request.instrumental_only:
        request.lyrics = "[instrumental]"
    
    # Генерация музыки
    audio_path, params_json = pipeline(
        format=request.format,
        audio_duration=request.audio_duration,
        prompt=prompt,
        lyrics=request.lyrics,
        infer_step=request.infer_step,
        guidance_scale=request.guidance_scale,
        guidance_interval=request.guidance_interval,
        guidance_interval_decay=request.guidance_interval_decay,
        min_guidance_scale=request.min_guidance_scale,
        use_erg_tag=request.use_erg_tag,
        use_erg_lyric=request.use_erg_lyric,
        use_erg_diffusion=request.use_erg_diffusion,
        cfg_type=request.cfg_type,
        scheduler_type=request.scheduler_type,
        omega_scale=request.omega_scale,
    )
    filename = os.path.basename(audio_path)
    audio_url = f"/static/{filename}"
    player_url = f"/audio_player/{filename}"
    return {
        "message": "Музыка успешно сгенерирована",
        "audio_url": audio_url,
        "player_url": player_url
    }

@app.get("/audio_player/{filename}", response_class=HTMLResponse)
async def audio_player(filename: str):
    audio_url = f"/static/{filename}"
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Аудиоплеер</title>
    </head>
    <body>
        <h1>Аудиоплеер</h1>
        <audio controls>
            <source src="{audio_url}" type="audio/{filename.split('.')[-1]}">
            Ваш браузер не поддерживает элемент audio.
        </audio>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)
