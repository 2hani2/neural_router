# app/main.py

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app.nanoqa import load_model
from app.router import route, calc as calc_expr

@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", "./model/nanoqa_v4_best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    load_model(path)
    print("[App] Ready")
    yield

app = FastAPI(title="Neural Router v4", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    try:
        result = route(req.question)
        if "error" in result:
            raise HTTPException(400, result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/calc")
async def calc_endpoint(expr: str):
    """/calc?expr=sqrt(144) — dedicated calculator, bypasses routing"""
    try:
        result, ms = calc_expr(expr)
        return {"expression": expr, "result": result, "latency_ms": ms}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
