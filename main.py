from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.database import engine, Base
from app.routers import tasks, priority, projects

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="FlowState API", version="1.0.0")

# CORS config for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use explicit origins e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tasks.router)
app.include_router(priority.router)
app.include_router(projects.router)

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "FlowState API"}
