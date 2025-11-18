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
    "Доктор": "Ночью лечит игрока, спасая его от устранения.",
    "Офицер": "Может арестовать игрока один раз за игру, блокируя его ход.",
    "Камикадзе": "При устранении забирает с собой одного мафиози.",
    "Фантом": "Появляется как мирный, но один раз может избежать голосования.",
    "Двойной агент": "Смотрит роль одного игрока и меняет сторону в зависимости от роли.",
}


# ---------- Pydantic-модели ----------

class HostRequest(BaseModel):
    slots: int
    roles: List[str]
    host_id: int
    host_name: str


class JoinRequest(BaseModel):
    user_id: int
    name: str


# ---------- FASTAPI ----------

app = FastAPI(title="Mafia Mini App API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # при желании потом ограничишь доменом фронта
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Mafia backend is running"}


@app.post("/api/games")
def host_game(body: HostRequest):
    """
    Создать игру.
    Ожидает:
    {
      "slots": 6,
      "roles": [...],
      "host_id": 123,
      "host_name": "Артур"
    }
    """
    if not 4 <= body.slots <= 12:
        raise HTTPException(status_code=400, detail="Количество мест должно быть от 4 до 12")

    allowed_roles = [role for role in body.roles if role in ROLE_DESCRIPTIONS]
    if not allowed_roles:
        allowed_roles = ["Мафия", "Детектив", "Доктор", "Мирный житель"]

    game = registry.create(host_id=body.host_id, slots=body.slots, allowed_roles=allowed_roles)

    # Хост сразу в лобби
    try:
        game.join(body.host_id, body.host_name)
    except ValueError:
        pass

    return {
        "code": game.code,
        "slots": game.slots,
        "roles": game.allowed_roles,
        "host_id": game.host_id,
        "started": game.started,
        "players": [{"user_id": p.user_id, "name": p.name} for p in game.players],
        "assignments": {},  # при создании игры ролей ещё нет
    }


@app.post("/api/games/{code}/join")
def join_game(code: str, body: JoinRequest):
    """
    Присоединиться к игре.
    Ожидает:
    { "user_id": 987, "name": "Игрок" }
    """
    try:
        game = registry.get(code)
        game.join(body.user_id, body.name)
        return {
            "status": "joined",
            "host_id": game.host_id,
            "players": [{"user_id": p.user_id, "name": p.name} for p in game.players],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/games/{code}/start")
def start_game(code: str):
    """
    Запустить игру: раздать роли всем игрокам.
    Возвращает assignments (словарь user_id -> роль).
    """
    try:
        game = registry.get(code)
        game.start()
        assignments = {str(uid): role for uid, role in game.assignments.items()}
        return {
            "status": "started",
            "assignments": assignments,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/games/{code}")
def get_game(code: str):
    """
    Получить текущее состояние игры / лобби.
    Если игра уже запущена (started = True), отдаём assignments,
    чтобы фронт мог показать роль текущего игрока.
    """
    try:
        game = registry.get(code)
        assignments = {str(uid): role for uid, role in game.assignments.items()} if game.started else {}
        return {
            "code": game.code,
            "slots": game.slots,
            "roles": game.allowed_roles,
            "host_id": game.host_id,
            "started": game.started,
            "players": [{"user_id": p.user_id, "name": p.name} for p in game.players],
            "assignments": assignments,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
