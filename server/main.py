from fastapi import FastAPI
from .route import router


app=FastAPI(version='1.0.0')

app.include_router(router)

@app.get('/')
def helth():
    return {"Status":"Runing"}






