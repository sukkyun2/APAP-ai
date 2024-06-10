from typing import Generic, TypeVar, Optional, List

from pydantic import BaseModel

T = TypeVar("T")


class ApiListResponse(BaseModel, Generic[T]):
    code: int = 200
    message: str = "OK"
    items: Optional[List[T]] = None

    @staticmethod
    def ok(items: List[T]):
        return ApiListResponse[T](code=200, message="OK", items=items)

    @staticmethod
    def bad_request(error: str):
        return ApiListResponse(code=400, message=error)

    class Config:
        arbitrary_types_allowed = True
