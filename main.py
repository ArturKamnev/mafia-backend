import os
import random
import string
import time
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==========
# OpenRouter (опционально, для ходов ботов через ИИ)
# ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
try:
    from openai import OpenAI  # pip install openai
    openrouter_client = (
        OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        if OPENROUTER_API_KEY
        else None
    )
except ImportError:
    openrouter_client = None


# ==========
# Модельки
# ==========

class Player(BaseModel):
    user_id: int
    name: str
    is_bot: bool = False
    alive: bool = True


class CreateGameRequest(BaseModel):
    slots: int = Field(ge=4, le=12)
    roles: List[str]
    host_id: int
    host_name: str


class JoinGameRequest(BaseModel):
    user_id: int
    name: str
    is_bot: bool = False


class ActionRequest(BaseModel):
    user_id: int
    action: str  # "kill" | "check" | "heal" | "vote"
    target_id: Optional[int] = None


class ChatMessageIn(BaseModel):
    user_id: int
    name: str
    text: str


# Структура игры в памяти (держим как dict)
GameState = Dict[str, Any]

games: Dict[str, GameState] = {}

# ==========
# FastAPI app
# ==========

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # при желании можешь ограничить доменом фронта
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========
# helpers
# ==========

def generate_code(length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    while True:
        code = "".join(random.choices(alphabet, k=length))
        if code not in games:
            return code


def game_summary(game: GameState) -> Dict[str, Any]:
    """То, что возвращаем фронту в /api/games и других местах."""
    return {
        "code": game["code"],
        "slots": game["slots"],
        "roles": game["roles"],
        "host_id": game["host_id"],
        "players": game["players"],
        "assignments": game.get("assignments", {}),
        "started": game.get("started", False),
        "phase": game.get("phase", "lobby"),
        "round": game.get("round", 1),
        "current_actor_id": game.get("current_actor_id"),
        "events": game.get("events", []),
    }


def get_game_or_404(code: str) -> GameState:
    game = games.get(code)
    if not game:
        raise HTTPException(status_code=404, detail="Игра не найдена")
    return game


def get_player(game: GameState, user_id: int) -> Player:
    for p in game["players"]:
        if p["user_id"] == user_id:
            return p
    raise HTTPException(status_code=404, detail="Игрок не найден в этой игре")


def get_alive_players(game: GameState) -> List[Player]:
    return [p for p in game["players"] if p.get("alive", True)]


def get_first_alive_with_role(game: GameState, role_name: str) -> Optional[Player]:
    assignments: Dict[str, str] = game.get("assignments", {})
    for uid_str, role in assignments.items():
        if role != role_name:
            continue
        uid = int(uid_str)
        for p in game["players"]:
            if p["user_id"] == uid and p.get("alive", True):
                return p
    return None


def ensure_basic_roles(roles: List[str]) -> List[str]:
    roles = roles[:]  # копия
    if "Мафия" not in roles:
        roles.append("Мафия")
    if "Мирный житель" not in roles:
        roles.append("Мирный житель")
    return roles


def assign_roles(game: GameState) -> Dict[str, str]:
    players = game["players"]
    roles_pool = ensure_basic_roles(game["roles"])
    assignments: Dict[str, str] = {}

    if not players:
        return assignments

    # гарантируем хотя бы одну мафию
    mafia_player = random.choice(players)
    assignments[str(mafia_player["user_id"])] = "Мафия"

    # остальным рандом по списку ролей
    for p in players:
        uid_str = str(p["user_id"])
        if uid_str in assignments:
            continue
        role = random.choice(roles_pool)
        assignments[uid_str] = role

    # гарантируем хотя бы одного мирного
    if "Мирный житель" not in assignments.values():
        non_mafia = [uid for uid, r in assignments.items() if r != "Мафия"]
        if non_mafia:
            uid_to_fix = random.choice(non_mafia)
            assignments[uid_to_fix] = "Мирный житель"

    return assignments


def start_night(game: GameState):
    """Перевод дня в ночь. Вызывается, например, из /bot-turn, когда фаза = day."""
    game["night_state"] = {
        "kill_target": None,
        "heal_target": None,
        "detective_target": None,
    }

    mafia = get_first_alive_with_role(game, "Мафия")
    detective = get_first_alive_with_role(game, "Детектив")
    doctor = get_first_alive_with_role(game, "Доктор")

    if mafia:
        game["phase"] = "night_mafia"
        game["current_actor_id"] = mafia["user_id"]
        return

    if detective:
        game["phase"] = "night_detective"
        game["current_actor_id"] = detective["user_id"]
        return

    if doctor:
        game["phase"] = "night_doctor"
        game["current_actor_id"] = doctor["user_id"]
        return

    # никого нет – ночь ничего не делает, сразу новый день
    resolve_night_and_go_day(game)


def goto_next_phase_after_mafia(game: GameState):
    detective = get_first_alive_with_role(game, "Детектив")
    doctor = get_first_alive_with_role(game, "Доктор")

    if detective:
        game["phase"] = "night_detective"
        game["current_actor_id"] = detective["user_id"]
    elif doctor:
        game["phase"] = "night_doctor"
        game["current_actor_id"] = doctor["user_id"]
    else:
        resolve_night_and_go_day(game)


def goto_next_phase_after_detective(game: GameState):
    doctor = get_first_alive_with_role(game, "Доктор")
    if doctor:
        game["phase"] = "night_doctor"
        game["current_actor_id"] = doctor["user_id"]
    else:
        resolve_night_and_go_day(game)


def resolve_night_and_go_day(game: GameState):
    """Рассчитываем итог ночи и переходим ко дню."""
    night_state = game.get("night_state", {})
    kill_target = night_state.get("kill_target")
    heal_target = night_state.get("heal_target")
    detective_target = night_state.get("detective_target")

    events: List[Dict[str, Any]] = []

    if detective_target is not None:
        # Можно добавить флаг is_mafia, если захочешь учесть это позже
        events.append({"type": "checked", "user_id": detective_target})

    if kill_target is not None:
        if heal_target == kill_target:
            events.append({"type": "healed", "user_id": kill_target})
        else:
            # убиваем игрока
            try:
                victim = get_player(game, kill_target)
                victim["alive"] = False
            except HTTPException:
                pass
            events.append({"type": "killed", "user_id": kill_target})

    game["events"] = events
    game["phase"] = "day"
    game["round"] = game.get("round", 1) + 1
    game["current_actor_id"] = None
    game["night_state"] = {}


def random_bot_action(game: GameState, bot_player: Player) -> Optional[Dict[str, Any]]:
    """Простейшее поведение бота, если нет OpenRouter."""
    phase = game.get("phase")
    alive_players = get_alive_players(game)
    # выбираем цели только среди живых, не самого себя
    candidates = [p for p in alive_players if p["user_id"] != bot_player["user_id"]]
    if not candidates:
        return None

    target = random.choice(candidates)
    if phase == "night_mafia":
        return {"action": "kill", "target_id": target["user_id"]}
    if phase == "night_detective":
        return {"action": "check", "target_id": target["user_id"]}
    if phase == "night_doctor":
        # доктор может лечить и самого себя, но для простоты иногда лечит себя, иногда другого
        if random.random() < 0.4:
            return {"action": "heal", "target_id": bot_player["user_id"]}
        return {"action": "heal", "target_id": target["user_id"]}

    return None


def build_ai_prompt_for_bot(game: GameState, bot_player: Player) -> str:
    """Промпт для OpenRouter: отдаём состояние и что хотим получить."""
    assignments: Dict[str, str] = game.get("assignments", {})
    role = assignments.get(str(bot_player["user_id"]), "Мирный житель")
    phase = game.get("phase")
    alive_players = get_alive_players(game)

    summary_players = [
        {
            "user_id": p["user_id"],
            "name": p["name"],
            "is_bot": p.get("is_bot", False),
            "alive": p.get("alive", True),
        }
        for p in alive_players
    ]

    return (
        "You are an AI agent playing the game Mafia.\n"
        f"Your role: {role}.\n"
        f"Current phase: {phase}.\n"
        "You see the list of alive players (including yourself):\n"
        f"{summary_players}\n\n"
        "Your task: choose exactly ONE action as a JSON object with keys 'action' and 'target_id'.\n"
        "Allowed actions:\n"
        "- if phase == 'night_mafia': action must be 'kill'.\n"
        "- if phase == 'night_detective': action must be 'check'.\n"
        "- if phase == 'night_doctor': action must be 'heal'.\n"
        "Choose any valid target_id from alive players.\n\n"
        "Return ONLY JSON, without explanations, like:\n"
        "{\"action\": \"kill\", \"target_id\": 123}\n"
    )


def ai_bot_action(game: GameState, bot_player: Player) -> Optional[Dict[str, Any]]:
    """Ход бота через OpenRouter; если не получилось – fallback на random_bot_action."""
    if not openrouter_client:
        return random_bot_action(game, bot_player)

    try:
        prompt = build_ai_prompt_for_bot(game, bot_player)
        completion = openrouter_client.chat.completions.create(
            model="moonshotai/kimi-k2:free",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        content = completion.choices[0].message.content
        # Пытаемся распарсить JSON, даже если модель вернула лишний текст
        import json
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            return random_bot_action(game, bot_player)
        obj = json.loads(content[start : end + 1])
        action = obj.get("action")
        target_id = obj.get("target_id")
        if action in ("kill", "check", "heal") and isinstance(target_id, int):
            return {"action": action, "target_id": target_id}
        return random_bot_action(game, bot_player)
    except Exception:
        # Любая ошибка – просто рандом
        return random_bot_action(game, bot_player)


# ==========
# Endpoints
# ==========

@app.post("/api/games")
def create_game(req: CreateGameRequest):
    code = generate_code()
    game: GameState = {
        "code": code,
        "slots": req.slots,
        "roles": req.roles or ["Мафия", "Мирный житель"],
        "host_id": req.host_id,
        "players": [
            {
                "user_id": req.host_id,
                "name": req.host_name,
                "is_bot": False,
                "alive": True,
            }
        ],
        "assignments": {},
        "started": False,
        "phase": "lobby",
        "round": 1,
        "current_actor_id": None,
        "events": [],
        "chat": [],
        "night_state": {},
    }
    games[code] = game
    return game_summary(game)


@app.post("/api/games/{code}/join")
def join_game(code: str, req: JoinGameRequest):
    game = get_game_or_404(code)

    if game.get("started"):
        raise HTTPException(status_code=400, detail="Игра уже началась")

    if len(game["players"]) >= game["slots"]:
        raise HTTPException(status_code=400, detail="Лобби заполнено")

    # если этот user_id уже есть – просто обновляем имя/флаг
    for p in game["players"]:
        if p["user_id"] == req.user_id:
            p["name"] = req.name
            p["is_bot"] = req.is_bot
            p["alive"] = True
            return game_summary(game)

    game["players"].append(
        {
            "user_id": req.user_id,
            "name": req.name,
            "is_bot": req.is_bot,
            "alive": True,
        }
    )

    return game_summary(game)


@app.get("/api/games/{code}")
def get_game(code: str):
    game = get_game_or_404(code)
    return game_summary(game)


@app.post("/api/games/{code}/start")
def start_game(code: str):
    game = get_game_or_404(code)

    if game.get("started"):
        # Идемпотентно – просто возвращаем текущее состояние
        return {
            "code": code,
            "assignments": game["assignments"],
            "phase": game["phase"],
            "round": game["round"],
            "events": game["events"],
            "players": game["players"],
            "host_id": game["host_id"],
            "started": True,
            "slots": game["slots"],
        }

    if len(game["players"]) < 4:
        raise HTTPException(status_code=400, detail="Нужно минимум 4 игрока")

    assignments = assign_roles(game)
    game["assignments"] = assignments
    game["started"] = True
    # Начинаем с дня (как ты хотел)
    game["phase"] = "day"
    game["round"] = 1
    game["current_actor_id"] = None
    game["events"] = []
    game["night_state"] = {}

    return {
        "code": code,
        "assignments": assignments,
        "phase": game["phase"],
        "round": game["round"],
        "events": game["events"],
        "players": game["players"],
        "host_id": game["host_id"],
        "started": True,
        "slots": game["slots"],
    }


@app.post("/api/games/{code}/action")
def game_action(code: str, req: ActionRequest):
    game = get_game_or_404(code)

    if not game.get("started"):
        raise HTTPException(status_code=400, detail="Игра ещё не началась")

    player = get_player(game, req.user_id)
    if not player.get("alive", True):
        raise HTTPException(status_code=400, detail="Мёртвые не ходят 🙂")

    phase = game.get("phase", "day")
    assignments: Dict[str, str] = game.get("assignments", {})
    role = assignments.get(str(req.user_id))

    if phase == "day":
        # Днём просто фиксим факт голосования (без логики вылета)
        if req.action == "vote" and req.target_id is not None:
            game["events"].append(
                {"type": "voted", "user_id": req.user_id, "target_id": req.target_id}
            )
        return {"ok": True}

    if phase == "night_mafia":
        if role != "Мафия":
            raise HTTPException(status_code=403, detail="Ход мафии, но вы не мафия")
        if game.get("current_actor_id") not in (None, req.user_id):
            raise HTTPException(status_code=403, detail="Сейчас ход другого игрока")
        if req.action != "kill" or req.target_id is None:
            raise HTTPException(status_code=400, detail="Ожидалось действие 'kill' с target_id")
        get_player(game, req.target_id)  # проверим, что цель существует
        # просто запоминаем цель, но не убиваем сейчас
        game.setdefault("night_state", {})["kill_target"] = req.target_id
        # после мафии идём к детективу/доктору/дню
        goto_next_phase_after_mafia(game)
        return {"ok": True}

    if phase == "night_detective":
        if role != "Детектив":
            raise HTTPException(status_code=403, detail="Ход детектива, но вы не детектив")
        if game.get("current_actor_id") not in (None, req.user_id):
            raise HTTPException(status_code=403, detail="Сейчас ход другого игрока")
        if req.action != "check" or req.target_id is None:
            raise HTTPException(status_code=400, detail="Ожидалось действие 'check' с target_id")

        get_player(game, req.target_id)
        game.setdefault("night_state", {})["detective_target"] = req.target_id
        goto_next_phase_after_detective(game)
        return {"ok": True}

    if phase == "night_doctor":
        if role != "Доктор":
            raise HTTPException(status_code=403, detail="Ход доктора, но вы не доктор")
        if game.get("current_actor_id") not in (None, req.user_id):
            raise HTTPException(status_code=403, detail="Сейчас ход другого игрока")
        if req.action != "heal" or req.target_id is None:
            raise HTTPException(status_code=400, detail="Ожидалось действие 'heal' с target_id")

        get_player(game, req.target_id)
        game.setdefault("night_state", {})["heal_target"] = req.target_id
        resolve_night_and_go_day(game)
        return {"ok": True}

    # На всякий случай
    raise HTTPException(status_code=400, detail=f"Неизвестная фаза: {phase}")


@app.get("/api/games/{code}/chat")
def get_chat(code: str):
    game = get_game_or_404(code)
    # возвращаем последние 100 сообщений
    chat = game.get("chat", [])
    return chat[-100:]


@app.post("/api/games/{code}/chat")
def post_chat(code: str, msg_in: ChatMessageIn):
    game = get_game_or_404(code)
    # определим, бот это или нет, по списку игроков
    is_bot = False
    try:
        p = get_player(game, msg_in.user_id)
        is_bot = bool(p.get("is_bot", False))
    except HTTPException:
        pass

    msg = {
        "user_id": msg_in.user_id,
        "name": msg_in.name,
        "text": msg_in.text,
        "ts": int(time.time() * 1000),
        "is_bot": is_bot,
    }
    game.setdefault("chat", []).append(msg)
    return {"ok": True}


@app.post("/api/games/{code}/bot-turn")
def bot_turn(code: str):
    """
    Кнопка «Сделать ход ботами» у хоста:
    - если сейчас day -> запускаем ночь (night_mafia / night_detective / night_doctor / сразу day);
    - если сейчас night_* -> пытаемся сделать ход тем ботом, у кого сейчас фаза.
    """
    game = get_game_or_404(code)

    if not game.get("started"):
        raise HTTPException(status_code=400, detail="Игра ещё не началась")

    phase = game.get("phase", "day")

    # если день – просто запускаем ночь
    if phase == "day":
        start_night(game)
        return game_summary(game)

    assignments: Dict[str, str] = game.get("assignments", {})
    current_actor_id = game.get("current_actor_id")

    # определим роль, у которой сейчас ход
    phase_role_map = {
        "night_mafia": "Мафия",
        "night_detective": "Детектив",
        "night_doctor": "Доктор",
    }
    role_needed = phase_role_map.get(phase)
    if not role_needed:
        return game_summary(game)

    # найдём бота с такой ролью
    bot_player: Optional[Player] = None
    for uid_str, role in assignments.items():
        if role != role_needed:
            continue
        uid = int(uid_str)
        for p in game["players"]:
            if p["user_id"] == uid and p.get("alive", True) and p.get("is_bot", False):
                bot_player = p
                break
        if bot_player:
            break

    if not bot_player:
        # нет бота для этой роли – ничего не делаем
        return game_summary(game)

    # если current_actor_id не совпадает – выставим его на бота
    game["current_actor_id"] = bot_player["user_id"]

    # получить действие от бота (ИИ или рандом)
    decision = ai_bot_action(game, bot_player)
    if not decision:
        return game_summary(game)

    action = decision["action"]
    target_id = decision["target_id"]

    # прогоняем через ту же логику, что и ручной ход
    _ = game_action(
        code,
        ActionRequest(
            user_id=bot_player["user_id"],
            action=action,
            target_id=target_id,
        ),
    )

    return game_summary(game)


# корневой маршрут, просто чтобы проверить, что сервер жив
@app.get("/")
def root():
    return {"status": "ok", "message": "Mafia backend is running"}
