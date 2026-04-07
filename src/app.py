from inference import load_model 
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import HTTPException 
from contextlib import asynccontextmanager 
from fastapi.responses import Response 


@asynccontextmanager 
async def lifespan(app: FastAPI): 
    try:
        app.state.classifier = load_model()
        print("Model loaded successfully")
        yield
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}") 
    finally:
        print("Application shutting down")
   
app = FastAPI(lifespan = lifespan, title="Tweet Classification")

        
class UserInput(BaseModel):
    text: str        

@app.get("/")
def root():
    return {"status": "ok", 
            "message": "Use POST /predict"}
    
@app.get("/health")
def health():
    return {"status": "ready to accept input"} 

@app.get("/favicon.ico")
async def get_favicon():
    return Response(status_code=200)

@app.post("/predict")
async def predict(payload: UserInput): 
    try: 
        output = app.state.classifier(payload.text)
        return output 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))