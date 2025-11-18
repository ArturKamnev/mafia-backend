import secrets
import string
from dataclasses import dataclass, field
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ---------- ВСПОМОГАТЕЛЬНОЕ ----------

def generate_code(length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@dataclass
class Player:
    user_id: int
    name: str


@dataclass
class Game:
    code: str
    host_id: int
    slots: int
    allowed_roles: List[str]
    started: bool = False
    players: List[Player] = field(default_factory=list)
    assignments: Dict[int, str] = field(default_factory=dict)

    def join(self, user_id: int, name: str) -> None:
        if self.started:
            raise ValueError("Игра уже началась")
        if any(p.user_id == user_id for p in self.players):
            raise ValueError("Вы уже в игре")
        if len(self.players) >= self.slots:
            raise ValueError("Все места заняты")
        self.players.append(Player(user_id=user_id, name=name))

    def start(self) -> None:
        if self.started:
            raise ValueError("Игра уже началась")
        if len(self.players) < 4:
            raise ValueError("Нужно минимум 4 игрока")
        self.started = True
        pool = self.allowed_roles.copy()
        while len(pool) < len(self.players):
            pool.append("Мирный житель")
        rng = secrets.SystemRandom()
        rng.shuffle(pool)
        for player, role in zip(self.players, pool):
            self.assignments[player.user_id] = role


class GameRegistry:
    def __init__(self) -> None:
        self.games: Dict[str, Game] = {}

    def create(self, host_id: int, slots: int, allowed_roles: List[str]) -> Game:
        code = generate_code()
        game = Game(code=code, host_id=host_id, slots=slots, allowed_roles=allowed_roles)
        self.games[code] = game
        return game

    def get(self, code: str) -> Game:
        try:
            return self.games[code]
        except KeyError:
            raise ValueError("Игра не найдена")


registry = GameRegistry()

ROLE_DESCRIPTIONS = {
    "Мирный житель": "Голосует днем, пытается вычислить мафию.",
    "Мафия": "Убирает игрока ночью. Цель — остаться в большинстве.",
    "Детектив": "Каждую ночь проверяет игрока и узнает его роль.",
    "Доктор": "Ночью лечит игрока
