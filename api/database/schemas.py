from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class TransactionBase(BaseModel):
    symbol: str
    type: str # BUY or SELL
    quantity: int
    price: float

class TransactionCreate(TransactionBase):
    id: str

class Transaction(TransactionBase):
    id: str
    timestamp: datetime
    class Config:
        orm_mode = True

class PositionBase(BaseModel):
    symbol: str
    quantity: int
    average_price: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None

class PositionCreate(PositionBase):
    pass

class Position(PositionBase):
    id: int
    class Config:
        orm_mode = True

class UserBase(BaseModel):
    username: Optional[str] = None
    email: str
    full_name: Optional[str] = None
    is_pro: bool = False

class UserCreate(UserBase):
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class User(UserBase):
    id: int
    cash_balance: float
    positions: List[Position] = []
    transactions: List[Transaction] = []
    class Config:
        from_attributes = True
