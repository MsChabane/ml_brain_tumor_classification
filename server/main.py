from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError 
from fastapi.middleware.cors import CORSMiddleware
from .route import router


app = FastAPI(version='1.0.0')


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    first_error = exc.errors()[0]
    field = first_error["loc"][-1]
    message = first_error["msg"]
    return JSONResponse(status_code=422, content={'detail':f"{field}: {message}"})

app.include_router(router)

@app.get('/')
def helth():
    return {"Status":"Runing"}






