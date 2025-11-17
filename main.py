import secrets
import string
from dataclasses import dataclass, field
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# ---------- МОДЕЛЬ ИГРЫ ----------

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
    "Доктор": "Ночью лечит игрока, спасая его от устранения.",
    "Офицер": "Может арестовать игрока один раз за игру, блокируя его ход.",
    "Камикадзе": "При устранении забирает с собой одного мафиози.",
    "Фантом": "Появляется как мирный, но один раз может избежать голосования.",
    "Двойной агент": "Смотрит роль одного игрока и меняет сторону в зависимости от роли.",
}

# ---------- FASTAPI ----------

class HostRequest(BaseModel):
    slots: int
    roles: List[str]


class JoinRequest(BaseModel):
    user_id: int
    name: str


app = FastAPI(title="Mafia Mini App API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # можно потом ограничить своим доменом
    allow_credentials=True,
    allow_methods=["*"],        # разрешаем любые методы (GET, POST, OPTIONS и т.д.)
    allow_headers=["*"],        # разрешаем любые заголовки
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Mafia backend is running"}


@app.post("/api/games")
def host_game(body: HostRequest):
    if not 4 <= body.slots <= 12:
        raise HTTPException(status_code=400, detail="Количество мест должно быть от 4 до 12")

    allowed_roles = [role for role in body.roles if role in ROLE_DESCRIPTIONS]
    if not allowed_roles:
        allowed_roles = ["Мафия", "Детектив", "Доктор", "Мирный житель"]

    game = registry.create(host_id=0, slots=body.slots, allowed_roles=allowed_roles)
    return {"code": game.code, "slots": game.slots, "roles": game.allowed_roles}


@app.post("/api/games/{code}/join")
def join_game(code: str, body: JoinRequest):
    try:
        game = registry.get(code)
        game.join(body.user_id, body.name)
        return {"status": "joined", "players": [p.name for p in game.players]}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/games/{code}/start")
def start_game(code: str):
    try:
        game = registry.get(code)
        game.start()
        return {
            "status": "started",
            "assignments": game.assignments,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/games/{code}")
def get_game(code: str):
    try:
        game = registry.get(code)
        return {
            "code": game.code,
            "players": [p.name for p in game.players],
            "started": game.started,
            "roles": game.allowed_roles,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
