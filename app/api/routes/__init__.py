# API routes module
from app.api.routes.classify import router as classify_router
from app.api.routes.health import router as health_router
from app.api.routes.training import router as training_router

__all__ = ["classify_router", "health_router", "training_router"]
