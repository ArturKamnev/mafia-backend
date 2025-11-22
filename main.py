import os
import json
import secrets
import string
from dataclasses import dataclass, field
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI  # библиотека openai v1

# ==============
#  OpenRouter client
# ==============

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MODEL_NAME = "moonshotai/kimi-k2:free"
ALLOWED_ACTIONS = ["vote", "kill", "heal", "check", "skip"]


# ==============
#  МОДЕЛЬ ИГРЫ
# ==============

def generate_code(length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@dataclass
class Player:
    user_id: int
    name: str
    is_bot: bool = False


@dataclass
class Game:
    code: str
    host_id: int
    slots: int
    allowed_roles: List[str]
    started: bool = False
    players: List[Player] = field(default_factory=list)
    assignments: Dict[int, str] = field(default_factory=dict)  # user_id -> role

    def join(self, user_id: int, name: str, is_bot: bool = False) -> None:
        if self.started:
            raise ValueError("Игра уже началась")
        if any(p.user_id == user_id for p in self.players):
            raise ValueError("Вы уже в игре")
        if len(self.players) >= self.slots:
            raise ValueError("Все места заняты")
        self.players.append(Player(user_id=user_id, name=name, is_bot=is_bot))

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
    "Мафия": "Убирает игроков ночью. Цель — остаться в большинстве.",
    "Детектив": "Каждую ночь проверяет игрока и узнает его роль.",
    "Доктор": "Ночью лечит игрока, спасая его от устранения.",
    "Офицер": "Может арестовать игрока один раз за игру, блокируя его ход.",
    "Камикадзе": "При устранении забирает с собой одного мафиози.",
    "Фантом": "Появляется как мирный, но один раз может избежать голосования.",
    "Двойной агент": "Смотрит роль одного игрока и меняет сторону в зависимости от роли.",
}


# ==============
#  Pydantic-схемы (под твой фронт)
# ==============

class HostRequest(BaseModel):
    slots: int
    roles: List[str]
    host_id: int
    host_name: str


class JoinRequest(BaseModel):
    user_id: int
    name: str
    is_bot: bool = False


class BotTurnRequest(BaseModel):
    user_id: int
    phase: str           # "day" или "night"
    history: List[dict] = []  # события/чат на будущее


# ==============
#  FastAPI app
# ==============

app = FastAPI(title="Mafia Mini App + OpenRouter AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # при желании можно ограничить доменом фронта
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Mafia backend with OpenRouter AI is running"}


# ==============
#  ЛОГИКА ДЛЯ AI-БОТОВ
# ==============

def serialize_game_state_for_ai(game: Game, acting_player_id: int, phase: str, history: List[dict]) -> dict:
    my_role = game.assignments.get(acting_player_id)
    return {
        "game_code": game.code,
        "phase": phase,
        "host_id": game.host_id,
        "started": game.started,
        "slots": game.slots,
        "roles_in_game": game.allowed_roles,
        "players": [
            {
                "user_id": p.user_id,
                "name": p.name,
                "is_bot": p.is_bot,
            }
            for p in game.players
        ],
        "acting_player_id": acting_player_id,
        "acting_player_role": my_role,
        "history": history,
    }


def build_ai_messages(game: Game, acting_player: Player, phase: str, history: List[dict]) -> List[dict]:
    state = serialize_game_state_for_ai(game, acting_player.user_id, phase, history)
    state_json = json.dumps(state, ensure_ascii=False, indent=2)

    system_content = (
        "Ты — игрок в настольной игре «Мафия». Ты управляешь одним персонажем.\n"
        "Тебе даётся текущее состояние игры в формате JSON: список игроков, фаза (день или ночь), "
        "возможные роли и твоя роль, а также история действий.\n\n"
        "ТВОЯ ЗАДАЧА — выбрать ОДНО действие строго в формате JSON.\n"
        "Формат ответа (строго один JSON-объект, без текста вокруг):\n"
        "{\n"
        '  \"action\": \"vote\" | \"kill\" | \"heal\" | \"check\" | \"skip\",\n'
        "  \"target_id\": число или null,\n"
        "  \"reason\": строка (1–2 предложения объяснения)\n"
        "}\n\n"
        "Правила:\n"
        "- Никакого текста вне JSON — ни приветствий, ни комментариев.\n"
        "- Если не можешь выбрать цель, используй action=\"skip\" и target_id=null.\n"
        "- Если у тебя роль мафии ночью — обычно action=\"kill\".\n"
        "- Если ты детектив ночью — action=\"check\".\n"
        "- Если ты доктор ночью — action=\"heal\".\n"
        "- Днём чаще всего используется action=\"vote\".\n"
    )

    user_content = (
        "Текущее состояние игры (JSON):\n"
        f"{state_json}\n\n"
        "Сгенерируй ОДНО действие в описанном формате JSON."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user",  "content": user_content},
    ]


def extract_json_from_text(text: str) -> dict:
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        raise ValueError("Модель не вернула корректный JSON.")
    json_str = text[first_brace:last_brace + 1]
    return json.loads(json_str)


def generate_ai_command(game: Game, acting_player: Player, phase: str, history: List[dict]) -> dict:
    messages = build_ai_messages(game, acting_player, phase, history)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        extra_headers={
            "HTTP-Referer": "https://projectmafia.onrender.com",
            "X-Title": "Mafia Mini App",
        },
    )

    text = completion.choices[0].message.content
    command = extract_json_from_text(text)

    action = command.get("action")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Недопустимое действие от модели: {action}")
    if "target_id" not in command:
        raise ValueError("В ответе ИИ нет поля target_id.")

    return command


# ==============
#  ЭНДПОИНТЫ ИГРЫ (под твой фронтенд)
# ==============

@app.post("/api/games")
def host_game(body: HostRequest):
    if not 4 <= body.slots <= 12:
        raise HTTPException(status_code=400, detail="Количество мест должно быть от 4 до 12")

    allowed_roles = [role for role in body.roles if role in ROLE_DESCRIPTIONS]
    if not allowed_roles:
        allowed_roles = ["Мафия", "Детектив", "Доктор", "Мирный житель"]

    game = registry.create(host_id=body.host_id, slots=body.slots, allowed_roles=allowed_roles)

    # хост сразу в лобби
    try:
        game.join(body.host_id, body.host_name, is_bot=False)
    except ValueError:
        pass

    return {
        "code": game.code,
        "slots": game.slots,
        "roles": game.allowed_roles,
        "host_id": game.host_id,
        "started": game.started,
        "players": [{"user_id": p.user_id, "name": p.name, "is_bot": p.is_bot} for p in game.players],
        "assignments": {},  # роли появятся после /start
    }


@app.post("/api/games/{code}/join")
def join_game(code: str, body: JoinRequest):
    try:
        game = registry.get(code)
        game.join(body.user_id, body.name, is_bot=body.is_bot)
        return {
            "status": "joined",
            "host_id": game.host_id,
            "players": [{"user_id": p.user_id, "name": p.name, "is_bot": p.is_bot} for p in game.players],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/games/{code}/start")
def start_game(code: str):
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
    try:
        game = registry.get(code)
        assignments = {str(uid): role for uid, role in game.assignments.items()} if game.started else {}
        return {
            "code": game.code,
            "slots": game.slots,
            "roles": game.allowed_roles,
            "host_id": game.host_id,
            "started": game.started,
            "players": [{"user_id": p.user_id, "name": p.name, "is_bot": p.is_bot} for p in game.players],
            "assignments": assignments,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/games/{code}/bot-turn")
def bot_turn(code: str, body: BotTurnRequest):
    """
    Ход ИИ-бота.
    Вход: user_id бота, phase ('day'/'night'), history (опционально).
    Выход: одна команда: { action, target_id, reason }.
    """
    try:
        game = registry.get(code)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    player = next((p for p in game.players if p.user_id == body.user_id), None)
    if not player:
        raise HTTPException(status_code=404, detail="Игрок не найден в этой игре")

    if not player.is_bot:
        raise HTTPException(status_code=400, detail="Этот игрок не помечен как бот (is_bot=false)")

    if not game.started:
        raise HTTPException(status_code=400, detail="Игра ещё не запущена")

    try:
        command = generate_ai_command(game, player, body.phase, body.history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации хода ИИ: {exc}")

    # Пока просто возвращаем команду, не применяя к состоянию игры
    return {"command": command}
