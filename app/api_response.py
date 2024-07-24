from typing import Generic, TypeVar, Optional, List

from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    code: int = 200
    message: str = "OK"
    item: Optional[T] = None

    @staticmethod
    def ok():
        return ApiResponse(code=200, message="OK")

    @staticmethod
    def ok_with_data(data: T):
        return ApiResponse(code=200, message="OK", item=data)

    @staticmethod
    def bad_request(error: str):
        return ApiResponse(code=400, message=error)

    class Config:
        arbitrary_types_allowed = True
