from contextlib import asynccontextmanager
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .config import APP_NAME, DEBUG
from .database import init_db
from .seed import seed_database
from .routers import shop, api, events, decision

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seeded = seed_database()
    if seeded:
        print("Database seeded with categories and products.")
    else:
        print("Database already populated, skipping seed.")
    yield


app = FastAPI(title=APP_NAME, debug=DEBUG, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

app.include_router(shop.router)
app.include_router(api.router)
app.include_router(events.router)
app.include_router(decision.router)
