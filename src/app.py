from inference import tweet_predictor 
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import HTTPException 
from contextlib import asynccontextmanager 
from fastapi.responses import Response 

app = FastAPI(title="Tweet Classification")

@asynccontextmanager 
async def warmup():
    try:
        if tweet_predictor: 
            print("warmup complete")
            
    except Exception as e:
        print("warmup failed", repr(e)) 
        
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
async def predict(text: UserInput): 
    try: 
        output = tweet_predictor(text.text)
        return output 
    except Exception as e:
        print("Prediction error", repr(e))
        raise HTTPException(status_code=500, details=str(e))