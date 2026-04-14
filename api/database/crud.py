from sqlalchemy.orm import Session
from . import models, schemas
import uuid

import hashlib

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = hash_password(user.password)
    db_user = models.User(
        email=user.email,
        full_name=user.full_name,
        is_pro=user.is_pro,
        password_hash=hashed_password,
        username=user.email.split("@")[0] # Simple username derivation
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_portfolio(db: Session, user_id: int):
    return db.query(models.PortfolioPosition).filter(models.PortfolioPosition.user_id == user_id).all()

def update_user_balance(db: Session, user_id: int, new_balance: float):
    user = get_user(db, user_id)
    if user:
        user.cash_balance = new_balance
        db.commit()
        db.refresh(user)
    return user

def execute_trade(db: Session, user_id: int, symbol: str, price: float, quantity: int, type: str, take_profit: float = None, stop_loss: float = None):
    # This acts as both BUY and SELL logic
    user = get_user(db, user_id)
    if not user:
        return None
        
    cost = price * quantity
    
    # Record transaction
    tx = models.TransactionHistory(
        id=str(uuid.uuid4())[:8],
        user_id=user_id,
        symbol=symbol,
        type=type,
        quantity=quantity,
        price=price
    )
    db.add(tx)
    
    # Update position
    pos = db.query(models.PortfolioPosition).filter(
        models.PortfolioPosition.user_id == user_id,
        models.PortfolioPosition.symbol == symbol
    ).first()
    
    if type == "BUY":
        user.cash_balance -= cost
        if pos:
            # weighted average
            total_cost = (pos.quantity * pos.average_price) + cost
            pos.quantity += quantity
            pos.average_price = total_cost / pos.quantity
            pos.take_profit = take_profit if take_profit else pos.take_profit
            pos.stop_loss = stop_loss if stop_loss else pos.stop_loss
        else:
            pos = models.PortfolioPosition(
                user_id=user_id,
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                take_profit=take_profit,
                stop_loss=stop_loss
            )
            db.add(pos)
    elif type == "SELL":
        user.cash_balance += cost
        if pos:
            pos.quantity -= quantity
            if pos.quantity <= 0:
                db.delete(pos)
    
    db.commit()
    return get_user(db, user_id)

def update_limits(db: Session, user_id: int, symbol: str, take_profit: float = None, stop_loss: float = None):
    pos = db.query(models.PortfolioPosition).filter(
        models.PortfolioPosition.user_id == user_id,
        models.PortfolioPosition.symbol == symbol
    ).first()
    
    if pos:
        pos.take_profit = take_profit
        pos.stop_loss = stop_loss
        db.commit()
        db.refresh(pos)
    return pos
