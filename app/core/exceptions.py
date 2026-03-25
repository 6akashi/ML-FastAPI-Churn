import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.errors.ErrorResponse import ErrorResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

def register_exception_handlers(app: FastAPI):
      @app.exception_handler(StarletteHTTPException)
      async def http_exception_handler(request: Request, exc: StarletteHTTPException):
           return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                     code="HTTP_ERROR",
                     message=exc.detail
                ).dict()
           )
      
      @app.exception_handler(RequestValidationError)
      async def validation_exception_handler(request: Request, exc: RequestValidationError):
          return JSONResponse(
              status_code=422,
              content=ErrorResponse(
                  code="VALIDATION_ERROR",
                  message="Error at data structure",
                  details=exc.errors()
              ).dict()
          )
      
      @app.exception_handler(Exception)
      async def common_exception_handler(request: Request, exc: Exception):
          return JSONResponse(
              status_code=500,
              content=ErrorResponse(
                  code="INTERNAL_SERVER_ERROR",
                  message="Произошла непредвиденная ошибка на сервере",
                  details=str(exc) if os.getenv("DEBUG") else None
              ).dict()
          )