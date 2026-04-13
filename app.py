import json
import os
import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import mysql.connector  # type: ignore
import mysql.connector.pooling  # type: ignore
import requests
from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover - dependency may not be installed yet
    genai = None


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

app = Flask(__name__)

def _env(key: str, default: str = "") -> str:
    value = os.getenv(key, default)
    return value.strip() if value else ""


def _env_first(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key, "")
        if value and value.strip():
            return value.strip()
    return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key, "true" if default else "false").lower()
    return raw in {"1", "true", "yes", "on"}


def sanitize_prefix(prefix: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "", prefix or "")
    if not cleaned:
        cleaned = "john_schedule_"
    if not cleaned.endswith("_"):
        cleaned += "_"
    return cleaned.lower()


def safe_next_path(raw: str | None) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    if not value.startswith("/") or value.startswith("//"):
        return ""
    return value
    
@app.route("/healthz")
def health():
    return "ok", 200

@dataclass(slots=True)
class Settings:
    secret_key: str = _env("FLASK_SECRET", secrets.token_hex(32))
    app_base_url: str = _env("APP_BASE_URL", "http://127.0.0.1:5001")
    app_timezone: str = _env("APP_TIMEZONE", "Africa/Nairobi")

    google_client_id: str = _env("GOOGLE_CLIENT_ID")
    google_client_secret: str = _env("GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = _env("GOOGLE_REDIRECT_URI")
    google_allowed_emails: str = _env_first("GOOGLE_ALLOWED_EMAILS", "ADMIN_GOOGLE_EMAILS")

    mysql_host: str = _env_first("MYSQL_HOST", "DB_HOST")
    mysql_port: int = int(_env_first("MYSQL_PORT", "DB_PORT", default="3306"))
    mysql_database: str = _env_first("MYSQL_DATABASE", "DB_NAME", "DB_DATABASE")
    mysql_user: str = _env_first("MYSQL_USER", "DB_USER")
    mysql_password: str = _env_first("MYSQL_PASSWORD", "DB_PASSWORD")
    mysql_ssl_ca: str = _env("MYSQL_SSL_CA")
    mysql_ssl_disabled: bool = _env_bool("MYSQL_SSL_DISABLED", default=False)
    mysql_connect_timeout: int = int(_env("MYSQL_CONNECT_TIMEOUT", "10"))
    mysql_use_pool: bool = _env_bool("MYSQL_USE_POOL", default=False)
    table_prefix: str = _env_first("SCHEDULE_TABLE_PREFIX", "DB_TABLE_PREFIX", default="john_schedule_")
    gemini_api_key: str = _env("GEMINI_API_KEY")
    gemini_model: str = _env("GEMINI_MODEL", "gemini-2.5-flash")
    ai_rate_limit_count: int = int(_env("AI_RATE_LIMIT_COUNT", "10"))
    ai_rate_limit_window_seconds: int = int(_env("AI_RATE_LIMIT_WINDOW_SECONDS", "60"))

    @property
    def mysql_enabled(self) -> bool:
        return all([self.mysql_host, self.mysql_database, self.mysql_user, self.mysql_password])

    @property
    def google_ready(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret and self.allowed_google_email_set)

    @property
    def allowed_google_email_set(self) -> set[str]:
        out: set[str] = set()
        for raw in (self.google_allowed_emails or "").split(","):
            email = raw.strip().lower()
            if email:
                out.add(email)
        return out

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key and genai is not None)


class MySQLStore:
    def __init__(self, settings: Settings) -> None:
        if not settings.mysql_enabled:
            raise RuntimeError("MySQL is not configured. Set MYSQL_HOST, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD.")
        self.prefix = sanitize_prefix(settings.table_prefix)
        self._connect_kwargs: dict[str, Any] = {
            "host": settings.mysql_host,
            "port": settings.mysql_port,
            "database": settings.mysql_database,
            "user": settings.mysql_user,
            "password": settings.mysql_password,
            "charset": "utf8mb4",
            "autocommit": False,
            "connection_timeout": settings.mysql_connect_timeout,
            "ssl_disabled": settings.mysql_ssl_disabled,
        }
        if not settings.mysql_ssl_disabled and settings.mysql_ssl_ca:
            self._connect_kwargs["ssl_ca"] = settings.mysql_ssl_ca
        self._pool = None
        if settings.mysql_use_pool:
            pool_kwargs = dict(self._connect_kwargs)
            pool_kwargs.update({"pool_name": "john_schedule_pool", "pool_size": 8, "pool_reset_session": False})
            self._pool = mysql.connector.pooling.MySQLConnectionPool(**pool_kwargs)

    def t(self, name: str) -> str:
        return f"{self.prefix}{re.sub(r'[^a-zA-Z0-9_]', '', name)}"

    def _connect(self):
        conn = self._pool.get_connection() if self._pool is not None else mysql.connector.connect(**self._connect_kwargs)
        conn.ping(reconnect=True, attempts=2, delay=0)
        return conn

    @staticmethod
    def _close(conn) -> None:
        try:
            conn.close()
        except Exception:
            pass

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(sql, params)
            return cur.fetchone()
        finally:
            self._close(conn)

    def query_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(sql, params)
            return list(cur.fetchall())
        finally:
            self._close(conn)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            conn.commit()
            return int(cur.lastrowid or 0)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._close(conn)

    def execute_many(self, statements: list[tuple[str, tuple[Any, ...]]]) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            for sql, params in statements:
                cur.execute(sql, params)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._close(conn)

    def ensure_schema(self) -> None:
        users = self.t("users")
        week_state = self.t("week_state")
        block_state = self.t("block_state")
        day_notes = self.t("day_notes")
        user_prefs = self.t("user_prefs")
        carry_forward = self.t("carry_forward")
        focus_block = self.t("focus_block")
        weekly_archive = self.t("weekly_archive")
        weekly_review = self.t("weekly_review")
        schedule_config = self.t("schedule_config")
        ai_weekly_insights = self.t("ai_weekly_insights")
        ai_compare_insights = self.t("ai_compare_insights")

        conn = self._connect()
        try:
            cur = conn.cursor()

            def try_exec(sql: str, params: tuple[Any, ...] = ()) -> None:
                try:
                    cur.execute(sql, params)
                except mysql.connector.Error as exc:
                    message = str(exc).lower()
                    benign = (
                        "duplicate column name" in message
                        or "duplicate key name" in message
                        or "already exists" in message
                        or "multiple primary key defined" in message
                    )
                    if not benign:
                        raise

            today = datetime.now(ZoneInfo(settings.app_timezone)).date()
            current_week = today.fromordinal(today.toordinal() - today.weekday()).isoformat()

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {users} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    google_sub VARCHAR(191) NOT NULL UNIQUE,
                    display_name VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            try_exec(f"ALTER TABLE {week_state} ADD COLUMN current_week_key VARCHAR(32) NOT NULL DEFAULT ''")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {week_state} (
                    user_id BIGINT PRIMARY KEY,
                    current_week_key VARCHAR(32) NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES {users}(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            try_exec(f"UPDATE {week_state} SET current_week_key = %s WHERE current_week_key = ''", (current_week,))

            try_exec(f"ALTER TABLE {block_state} ADD COLUMN week_key VARCHAR(32) NOT NULL DEFAULT '' AFTER user_id")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {block_state} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    week_key VARCHAR(32) NOT NULL,
                    day_key VARCHAR(64) NOT NULL,
                    block_idx INT NOT NULL,
                    completed TINYINT(1) NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uniq_user_week_day_block (user_id, week_key, day_key, block_idx),
                    FOREIGN KEY (user_id) REFERENCES {users}(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            try_exec(f"UPDATE {block_state} SET week_key = %s WHERE week_key = ''", (current_week,))
            try_exec(f"ALTER TABLE {block_state} ADD UNIQUE KEY uniq_user_week_day_block (user_id, week_key, day_key, block_idx)")

            try_exec(f"ALTER TABLE {day_notes} ADD COLUMN week_key VARCHAR(32) NOT NULL DEFAULT '' AFTER user_id")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {day_notes} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    week_key VARCHAR(32) NOT NULL,
                    day_key VARCHAR(64) NOT NULL,
                    note_text MEDIUMTEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uniq_user_week_day_note (user_id, week_key, day_key),
                    FOREIGN KEY (user_id) REFERENCES {users}(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            try_exec(f"UPDATE {day_notes} SET week_key = %s WHERE week_key = ''", (current_week,))
            try_exec(f"ALTER TABLE {day_notes} ADD UNIQUE KEY uniq_user_week_day_note (user_id, week_key, day_key)")

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {user_prefs} (
                    user_id BIGINT PRIMARY KEY,
                    compact_mode TINYINT(1) NOT NULL DEFAULT 0,
                    hide_completed TINYINT(1) NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES {users}(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {carry_forward} (
                    user_id BIGINT PRIMARY KEY,
                    note_text MEDIUMTEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES {users}(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            try_exec(f"ALTER TABLE {focus_block} ADD COLUMN week_key VARCHAR(32) NOT NULL DEFAULT '' AFTER user_id")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {focus_block} (
                    user_id BIGINT NOT NULL,
                    week_key VARCHAR(32) NOT NULL,
                    day_key VARCHAR(64) NOT NULL,
                    block_idx INT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, week_key, day_key),
                    FOREIGN KEY (user_id) REFERENCES {users}(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            try_exec(f"UPDATE {focus_block} SET week_key = %s WHERE week_key = ''", (current_week,))
            try_exec(f"ALTER TABLE {focus_block} ADD UNIQUE KEY uniq_focus_week_day (user_id, week_key, day_key)")

            try_exec(f"ALTER TABLE {weekly_archive} ADD COLUMN week_key VARCHAR(32) NOT NULL DEFAULT '' AFTER user_id")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {weekly_archive} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    week_key VARCHAR(32) NOT NULL,
                    completed_blocks INT NOT NULL DEFAULT 0,
                    total_blocks INT NOT NULL DEFAULT 0,
                    carry_forward_note MEDIUMTEXT NOT NULL,
                    snapshot_json LONGTEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_archive_user_created (user_id, created_at),
                    FOREIGN KEY (user_id) REFERENCES {users}(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            try_exec(
                f"""
                UPDATE {weekly_archive}
                SET week_key = DATE_FORMAT(DATE_SUB(DATE(created_at), INTERVAL WEEKDAY(DATE(created_at)) DAY), '%Y-%m-%d')
                WHERE week_key = ''
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {weekly_review} (
                    user_id BIGINT PRIMARY KEY,
                    wins_text MEDIUMTEXT NOT NULL,
                    misses_text MEDIUMTEXT NOT NULL,
                    adjust_text MEDIUMTEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES {users}(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {schedule_config} (
                    id TINYINT PRIMARY KEY,
                    schedule_json LONGTEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {ai_weekly_insights} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    archive_id BIGINT NOT NULL,
                    week_key VARCHAR(32) NOT NULL,
                    model_name VARCHAR(120) NOT NULL,
                    context_json LONGTEXT NOT NULL,
                    analysis_json LONGTEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uniq_user_archive_insight (user_id, archive_id),
                    FOREIGN KEY (user_id) REFERENCES {users}(id) ON DELETE CASCADE,
                    FOREIGN KEY (archive_id) REFERENCES {weekly_archive}(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {ai_compare_insights} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    anchor_archive_id BIGINT NOT NULL,
                    week_count INT NOT NULL,
                    model_name VARCHAR(120) NOT NULL,
                    context_json LONGTEXT NOT NULL,
                    analysis_json LONGTEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uniq_user_compare_anchor (user_id, anchor_archive_id, week_count),
                    FOREIGN KEY (user_id) REFERENCES {users}(id) ON DELETE CASCADE,
                    FOREIGN KEY (anchor_archive_id) REFERENCES {weekly_archive}(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._close(conn)


def validate_schedule(payload: dict[str, Any]) -> dict[str, Any]:
    days = payload.get("days") or []
    seen: set[str] = set()
    if not days:
        raise RuntimeError("Schedule JSON is empty.")
    for day in days:
        key = str(day.get("key") or "").strip()
        if not key or key in seen:
            raise RuntimeError("Each day in schedule JSON must have a unique key.")
        seen.add(key)
        blocks = day.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise RuntimeError(f"Day '{key}' must contain blocks.")
    return payload


def load_schedule_file() -> dict[str, Any]:
    return validate_schedule(json.loads((ROOT / "data" / "schedule.json").read_text(encoding="utf-8")))


settings = Settings()
store = MySQLStore(settings)
store.ensure_schema()
default_schedule = load_schedule_file()
if settings.gemini_enabled:
    genai.configure(api_key=settings.gemini_api_key)


def get_schedule_data() -> dict[str, Any]:
    row = store.query_one(f"SELECT schedule_json FROM {store.t('schedule_config')} WHERE id = 1")
    if row and row.get("schedule_json"):
        try:
            return validate_schedule(json.loads(str(row["schedule_json"])))
        except Exception:
            pass
    return default_schedule


app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = settings.secret_key
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = settings.app_base_url.lower().startswith("https://")


def csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return str(token)


def verify_csrf() -> bool:
    expected = session.get("csrf_token")
    provided = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
    return bool(expected and provided and str(expected) == str(provided))


def clear_auth_session() -> None:
    for key in ("user_id", "email", "display_name", "google_sub", "auth_provider", "post_auth_redirect", "google_oauth_state"):
        session.pop(key, None)


def session_user() -> dict[str, Any] | None:
    user_id = session.get("user_id")
    email = (session.get("email") or "").strip().lower()
    google_sub = (session.get("google_sub") or "").strip()
    provider = (session.get("auth_provider") or "").strip().lower()
    if not user_id or not email or not google_sub or provider != "google":
        return None
    if email not in settings.allowed_google_email_set:
        return None
    return {"id": int(user_id), "email": email, "display_name": (session.get("display_name") or email).strip(), "google_sub": google_sub}


def login_required(fn):
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        user = session_user()
        if not user:
            clear_auth_session()
            if request.path.startswith("/api/"):
                return jsonify({"ok": False, "error": "Authentication required."}), 401
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)

    return wrapper


def now_in_app_tz() -> datetime:
    return datetime.now(ZoneInfo(settings.app_timezone))


def current_week_key() -> str:
    today = now_in_app_tz().date()
    monday = today.fromordinal(today.toordinal() - today.weekday())
    return monday.isoformat()


def week_label_from_key(week_key: str) -> str:
    monday = datetime.fromisoformat(week_key).date()
    sunday = monday.fromordinal(monday.toordinal() + 6)
    if os.name == "nt":
        return f"Week of {monday.strftime('%B')} {monday.day}, {monday.year} to {sunday.strftime('%B')} {sunday.day}, {sunday.year}"
    return f"Week of {monday.strftime('%B %-d, %Y')} to {sunday.strftime('%B %-d, %Y')}"


def build_user_snapshot(user_id: int, week_key: str, schedule_data: dict[str, Any]) -> dict[str, Any]:
    block_rows = store.query_all(
        f"SELECT day_key, block_idx, completed FROM {store.t('block_state')} WHERE user_id = %s AND week_key = %s",
        (user_id, week_key),
    )
    note_rows = store.query_all(
        f"SELECT day_key, note_text FROM {store.t('day_notes')} WHERE user_id = %s AND week_key = %s",
        (user_id, week_key),
    )
    focus_rows = store.query_all(
        f"SELECT day_key, block_idx FROM {store.t('focus_block')} WHERE user_id = %s AND week_key = %s",
        (user_id, week_key),
    )
    pref_row = store.query_one(f"SELECT compact_mode, hide_completed FROM {store.t('user_prefs')} WHERE user_id = %s", (user_id,)) or {
        "compact_mode": 0,
        "hide_completed": 0,
    }
    carry_row = store.query_one(f"SELECT note_text FROM {store.t('carry_forward')} WHERE user_id = %s", (user_id,)) or {"note_text": ""}
    review_row = store.query_one(
        f"SELECT wins_text, misses_text, adjust_text FROM {store.t('weekly_review')} WHERE user_id = %s",
        (user_id,),
    ) or {
        "wins_text": "",
        "misses_text": "",
        "adjust_text": "",
    }

    blocks: dict[str, dict[str, bool]] = {}
    for row in block_rows:
        blocks.setdefault(str(row["day_key"]), {})[str(int(row["block_idx"]))] = bool(row["completed"])

    notes = {str(row["day_key"]): str(row["note_text"] or "") for row in note_rows}
    focus = {str(row["day_key"]): int(row["block_idx"]) for row in focus_rows}

    total_blocks = sum(len(day["blocks"]) for day in schedule_data["days"])
    completed_blocks = sum(
        1
        for day in schedule_data["days"]
        for index, _ in enumerate(day["blocks"])
        if bool((blocks.get(day["key"]) or {}).get(str(index)))
    )

    return {
        "week_key": week_key,
        "week_label": week_label_from_key(week_key),
        "schedule": schedule_data,
        "blocks": blocks,
        "notes": notes,
        "focus": focus,
        "preferences": {
            "compact_mode": bool(pref_row.get("compact_mode")),
            "hide_completed": bool(pref_row.get("hide_completed")),
        },
        "carry_forward_note": str(carry_row.get("note_text") or ""),
        "weekly_review": {
            "wins_text": str(review_row.get("wins_text") or ""),
            "misses_text": str(review_row.get("misses_text") or ""),
            "adjust_text": str(review_row.get("adjust_text") or ""),
        },
        "completed_blocks": completed_blocks,
        "total_blocks": total_blocks,
    }


def review_has_content(review_data: dict[str, Any]) -> bool:
    return bool(
        str(review_data.get("wins_text") or "").strip()
        or str(review_data.get("misses_text") or "").strip()
        or str(review_data.get("adjust_text") or "").strip()
    )


def archive_view_model(snapshot: dict[str, Any]) -> dict[str, Any]:
    if isinstance(snapshot.get("state"), dict):
        state = dict(snapshot["state"])
        days = list(snapshot.get("days") or [])
        schedule_data = {"days": days}
    else:
        schedule_data = snapshot.get("schedule") if isinstance(snapshot.get("schedule"), dict) else {}
        days = list(schedule_data.get("days") or [])
        state = {
            "blocks": snapshot.get("blocks") or {},
            "notes": snapshot.get("notes") or {},
            "focus": snapshot.get("focus") or {},
            "carry_forward_note": snapshot.get("carry_forward_note") or "",
            "weekly_review": snapshot.get("weekly_review") or {},
            "completed_blocks": snapshot.get("completed_blocks") or 0,
            "total_blocks": snapshot.get("total_blocks") or sum(len(day.get("blocks") or []) for day in days),
        }

    state.setdefault("blocks", {})
    state.setdefault("notes", {})
    state.setdefault("focus", {})
    state.setdefault("carry_forward_note", "")
    review_state = state.get("weekly_review")
    if not isinstance(review_state, dict):
        review_state = {}
    review_state.setdefault("wins_text", "")
    review_state.setdefault("misses_text", "")
    review_state.setdefault("adjust_text", "")
    state["weekly_review"] = review_state
    state.setdefault("completed_blocks", 0)
    state.setdefault("total_blocks", sum(len(day.get("blocks") or []) for day in days))

    total_hours = 0.0
    deep_work_hours = 0.0
    track_stats: dict[str, dict[str, int]] = {
        "ml_project": {"done": 0, "total": 0},
        "stats": {"done": 0, "total": 0},
        "cpp": {"done": 0, "total": 0},
    }
    blocks_state = state["blocks"] if isinstance(state["blocks"], dict) else {}

    for day in days:
        day_key = str(day.get("key") or "")
        saved_day = blocks_state.get(day_key, {})
        if not isinstance(saved_day, dict):
            saved_day = {}
        for index, block in enumerate(day.get("blocks") or []):
            hours = float(block.get("hours") or 0)
            total_hours += hours
            if str(block.get("cat") or "") != "health":
                deep_work_hours += hours

            track = str(block.get("track") or "")
            if track in track_stats:
                track_stats[track]["total"] += 1
                if bool(saved_day.get(str(index))):
                    track_stats[track]["done"] += 1

    return {
        "days": days,
        "state": state,
        "schedule": schedule_data,
        "stats": {
            "total_hours": round(total_hours, 1),
            "deep_work_hours": round(deep_work_hours, 1),
            "ml_project": track_stats["ml_project"],
            "stats": track_stats["stats"],
            "cpp": track_stats["cpp"],
        },
    }


def clip_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 3].rstrip()
    return f"{clipped}..."


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = (raw_text or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def normalize_insight_payload(payload: dict[str, Any] | None, raw_text: str = "") -> dict[str, Any]:
    base = payload if isinstance(payload, dict) else {}

    def string_list(key: str, max_items: int = 6, max_chars: int = 220) -> list[str]:
        values = base.get(key)
        if not isinstance(values, list):
            return []
        out: list[str] = []
        for item in values[:max_items]:
            text = clip_text(item, max_chars)
            if text:
                out.append(text)
        return out

    overview = clip_text(base.get("week_overview") or raw_text or "No AI overview available yet.", 600)
    keep_rule = clip_text(base.get("keep_rule"), 220)
    change_rule = clip_text(base.get("change_rule"), 220)
    confidence_note = clip_text(base.get("confidence_note"), 220)

    return {
        "week_overview": overview,
        "what_worked": string_list("what_worked"),
        "what_did_not_work": string_list("what_did_not_work"),
        "patterns": string_list("patterns"),
        "recommendations_next_week": string_list("recommendations_next_week"),
        "keep_rule": keep_rule,
        "change_rule": change_rule,
        "confidence_note": confidence_note,
        "raw_text": clip_text(raw_text, 4000),
    }


def normalize_compare_payload(payload: dict[str, Any] | None, raw_text: str = "") -> dict[str, Any]:
    base = payload if isinstance(payload, dict) else {}

    def string_list(key: str, max_items: int = 8, max_chars: int = 220) -> list[str]:
        values = base.get(key)
        if not isinstance(values, list):
            return []
        out: list[str] = []
        for item in values[:max_items]:
            text = clip_text(item, max_chars)
            if text:
                out.append(text)
        return out

    return {
        "summary": clip_text(base.get("summary") or raw_text or "No AI comparison available yet.", 700),
        "recurring_strengths": string_list("recurring_strengths"),
        "recurring_breakdowns": string_list("recurring_breakdowns"),
        "patterns_across_weeks": string_list("patterns_across_weeks"),
        "next_week_adjustments": string_list("next_week_adjustments"),
        "keep_doing": clip_text(base.get("keep_doing"), 220),
        "stop_doing": clip_text(base.get("stop_doing"), 220),
        "watch_item": clip_text(base.get("watch_item"), 220),
        "raw_text": clip_text(raw_text, 4000),
    }


def build_ai_week_context(archive: dict[str, Any], archive_view: dict[str, Any]) -> dict[str, Any]:
    day_summaries: list[dict[str, Any]] = []
    notes = archive_view["state"].get("notes", {})
    blocks_state = archive_view["state"].get("blocks", {})
    focus_state = archive_view["state"].get("focus", {})

    for day in archive_view["days"]:
        day_key = str(day.get("key") or "")
        saved_blocks = blocks_state.get(day_key, {}) if isinstance(blocks_state, dict) else {}
        if not isinstance(saved_blocks, dict):
            saved_blocks = {}
        blocks: list[dict[str, Any]] = []
        for index, block in enumerate(day.get("blocks") or []):
            blocks.append(
                {
                    "title": clip_text(block.get("title"), 120),
                    "category": clip_text(block.get("cat"), 40),
                    "hours": float(block.get("hours") or 0),
                    "done": bool(saved_blocks.get(str(index))),
                    "focused": int(focus_state.get(day_key, -1)) == index,
                }
            )

        day_summaries.append(
            {
                "day": clip_text(day.get("name"), 40),
                "focus": clip_text(day.get("focus"), 180),
                "note": clip_text(notes.get(day_key, ""), 500),
                "blocks": blocks,
            }
        )

    return {
        "week_key": archive.get("week_key"),
        "week_label": week_label_from_key(str(archive.get("week_key") or "")),
        "stats": archive_view["stats"],
        "completion": {
            "completed_blocks": archive_view["state"].get("completed_blocks", 0),
            "total_blocks": archive_view["state"].get("total_blocks", 0),
        },
        "carry_forward_note": clip_text(archive_view["state"].get("carry_forward_note", ""), 600),
        "weekly_review": {
            "wins": clip_text(archive_view["state"].get("weekly_review", {}).get("wins_text", ""), 900),
            "misses": clip_text(archive_view["state"].get("weekly_review", {}).get("misses_text", ""), 900),
            "adjustments": clip_text(archive_view["state"].get("weekly_review", {}).get("adjust_text", ""), 900),
        },
        "days": day_summaries,
    }


def build_compare_context(archives: list[dict[str, Any]]) -> dict[str, Any]:
    weeks: list[dict[str, Any]] = []
    for archive in archives:
        snapshot = json.loads(str(archive["snapshot_json"]))
        archive_view = archive_view_model(snapshot)
        weekly_review = archive_view["state"].get("weekly_review", {})
        weeks.append(
            {
                "archive_id": int(archive["id"]),
                "week_key": str(archive.get("week_key") or ""),
                "week_label": week_label_from_key(str(archive.get("week_key") or "")),
                "stats": archive_view["stats"],
                "completion": {
                    "completed_blocks": archive_view["state"].get("completed_blocks", 0),
                    "total_blocks": archive_view["state"].get("total_blocks", 0),
                },
                "carry_forward_note": clip_text(archive_view["state"].get("carry_forward_note", ""), 450),
                "weekly_review": {
                    "wins": clip_text(weekly_review.get("wins_text", ""), 600),
                    "misses": clip_text(weekly_review.get("misses_text", ""), 600),
                    "adjustments": clip_text(weekly_review.get("adjust_text", ""), 600),
                },
                "day_notes": [
                    {
                        "day": clip_text(day.get("name"), 40),
                        "note": clip_text((archive_view["state"].get("notes", {}) or {}).get(str(day.get("key") or ""), ""), 260),
                    }
                    for day in archive_view["days"]
                    if clip_text((archive_view["state"].get("notes", {}) or {}).get(str(day.get("key") or ""), ""), 260)
                ],
            }
        )
    return {
        "comparison_window": f"Latest {len(weeks)} archived weeks",
        "weeks": weeks,
    }


def build_weekly_ai_prompt(context: dict[str, Any]) -> str:
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return f"""You are an analytical weekly planning coach.

Your job is to analyze one completed week and suggest realistic improvements for the following week.

Rules:
1. Use only the supplied weekly data.
2. Be specific and practical.
3. Do not recommend a complete life overhaul.
4. Respect the user's existing system unless the data clearly shows a recurring issue.
5. Return strict JSON only. No markdown. No code fences.

Return this JSON shape exactly:
{{
  "week_overview": "2-4 sentence summary",
  "what_worked": ["short point", "short point"],
  "what_did_not_work": ["short point", "short point"],
  "patterns": ["short point", "short point"],
  "recommendations_next_week": ["specific recommendation", "specific recommendation", "specific recommendation"],
  "keep_rule": "one rule to preserve",
  "change_rule": "one rule to change next week",
  "confidence_note": "short note about confidence or missing context"
}}

Weekly data:
{context_json}
"""


def build_compare_ai_prompt(context: dict[str, Any]) -> str:
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return f"""You are an analytical weekly planning coach.

Your job is to compare multiple completed weeks and identify useful patterns for the next week.

Rules:
1. Use only the supplied weekly data.
2. Focus on recurring signals, not one-off noise.
3. Make recommendations that preserve the user's overall system unless there is strong evidence it is failing.
4. Return strict JSON only. No markdown. No code fences.

Return this JSON shape exactly:
{{
  "summary": "3-5 sentence summary of the multi-week pattern",
  "recurring_strengths": ["short point", "short point"],
  "recurring_breakdowns": ["short point", "short point"],
  "patterns_across_weeks": ["pattern", "pattern"],
  "next_week_adjustments": ["specific adjustment", "specific adjustment", "specific adjustment"],
  "keep_doing": "one thing the user should keep doing",
  "stop_doing": "one thing the user should stop doing",
  "watch_item": "one thing to monitor next week"
}}

Weekly comparison data:
{context_json}
"""


def check_ai_rate_limit() -> str | None:
    now = time.time()
    count_key = "ai_weekly_request_count"
    reset_key = "ai_weekly_reset_at"
    if count_key not in session:
        session[count_key] = 0
        session[reset_key] = now
    if now - float(session.get(reset_key, now)) > settings.ai_rate_limit_window_seconds:
        session[count_key] = 0
        session[reset_key] = now
    if int(session.get(count_key, 0)) >= settings.ai_rate_limit_count:
        return "AI weekly analysis rate limit reached. Please wait a few minutes and try again."
    session[count_key] = int(session.get(count_key, 0)) + 1
    return None


def generate_weekly_ai_analysis(context: dict[str, Any]) -> dict[str, Any]:
    if not settings.gemini_enabled:
        raise RuntimeError("Gemini AI is not configured yet.")
    prompt = build_weekly_ai_prompt(context)
    model = genai.GenerativeModel(settings.gemini_model)
    response = model.generate_content(prompt)
    raw_text = str(getattr(response, "text", "") or "").strip()
    parsed = extract_json_object(raw_text)
    return normalize_insight_payload(parsed, raw_text)


def generate_compare_ai_analysis(context: dict[str, Any]) -> dict[str, Any]:
    if not settings.gemini_enabled:
        raise RuntimeError("Gemini AI is not configured yet.")
    prompt = build_compare_ai_prompt(context)
    model = genai.GenerativeModel(settings.gemini_model)
    response = model.generate_content(prompt)
    raw_text = str(getattr(response, "text", "") or "").strip()
    parsed = extract_json_object(raw_text)
    return normalize_compare_payload(parsed, raw_text)


def ensure_user_week(user_id: int) -> str:
    wk = current_week_key()
    week_state_table = store.t("week_state")
    row = store.query_one(f"SELECT current_week_key FROM {week_state_table} WHERE user_id = %s", (user_id,))
    if not row:
        store.execute(f"INSERT INTO {week_state_table} (user_id, current_week_key) VALUES (%s, %s)", (user_id, wk))
        return wk

    previous = str(row["current_week_key"])
    if previous == wk:
        return wk

    schedule_data = get_schedule_data()
    snapshot = build_user_snapshot(user_id, previous, schedule_data)
    has_activity = (
        snapshot["completed_blocks"] > 0
        or any(snapshot["notes"].values())
        or bool(snapshot["focus"])
        or bool(snapshot["carry_forward_note"])
        or review_has_content(snapshot["weekly_review"])
    )
    statements: list[tuple[str, tuple[Any, ...]]] = []
    if has_activity:
        statements.append(
            (
                f"""
                INSERT INTO {store.t('weekly_archive')} (user_id, week_key, completed_blocks, total_blocks, carry_forward_note, snapshot_json)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    previous,
                    int(snapshot["completed_blocks"]),
                    int(snapshot["total_blocks"]),
                    snapshot["carry_forward_note"],
                    json.dumps(snapshot, ensure_ascii=False),
                ),
            )
        )
    statements.extend(
        [
            (f"DELETE FROM {store.t('block_state')} WHERE user_id = %s AND week_key = %s", (user_id, previous)),
            (f"DELETE FROM {store.t('day_notes')} WHERE user_id = %s AND week_key = %s", (user_id, previous)),
            (f"DELETE FROM {store.t('focus_block')} WHERE user_id = %s AND week_key = %s", (user_id, previous)),
            (f"DELETE FROM {store.t('weekly_review')} WHERE user_id = %s", (user_id,)),
            (f"UPDATE {week_state_table} SET current_week_key = %s WHERE user_id = %s", (wk, user_id)),
        ]
    )
    store.execute_many(statements)
    return wk


def _day_lookup() -> dict[str, Any]:
    schedule_data = get_schedule_data()
    return {day["key"]: day for day in schedule_data["days"]}


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})


@app.get("/login")
def login():
    if session_user():
        return redirect(url_for("index"))
    return render_template("login.html", google_ready=settings.google_ready, allowed_emails=sorted(settings.allowed_google_email_set))


@app.post("/logout")
def logout():
    if not verify_csrf():
        return "Invalid CSRF token.", 400
    session.clear()
    return redirect(url_for("login"))


@app.get("/auth/google/start")
def google_start():
    if not settings.google_client_id or not settings.google_client_secret:
        flash("Google OAuth is not configured yet.", "error")
        return redirect(url_for("login"))
    if not settings.allowed_google_email_set:
        flash("No Google email allowlist found in .env.", "error")
        return redirect(url_for("login"))

    state = secrets.token_urlsafe(24)
    session["google_oauth_state"] = state
    session["post_auth_redirect"] = safe_next_path(request.args.get("next")) or url_for("index")
    redirect_uri = settings.google_redirect_uri or url_for("google_callback", _external=True)
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "offline",
        "prompt": "select_account",
    }
    return redirect(f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}")


@app.get("/auth/google/callback")
def google_callback():
    next_path = safe_next_path(session.pop("post_auth_redirect", "")) or url_for("index")
    expected_state = session.pop("google_oauth_state", None)
    if not expected_state or request.args.get("state") != expected_state:
        flash("Google auth state mismatch.", "error")
        return redirect(url_for("login"))

    code = request.args.get("code")
    if not code:
        flash("Google did not return an authorization code.", "error")
        return redirect(url_for("login"))

    redirect_uri = settings.google_redirect_uri or url_for("google_callback", _external=True)
    token_resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=10,
    )
    if token_resp.status_code >= 300:
        flash("Google token exchange failed.", "error")
        return redirect(url_for("login"))

    access_token = (token_resp.json() or {}).get("access_token")
    if not access_token:
        flash("Google did not return an access token.", "error")
        return redirect(url_for("login"))

    profile_resp = requests.get(
        "https://openidconnect.googleapis.com/v1/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    if profile_resp.status_code >= 300:
        flash("Google profile lookup failed.", "error")
        return redirect(url_for("login"))

    profile = profile_resp.json() or {}
    google_sub = (profile.get("sub") or "").strip()
    email = (profile.get("email") or "").strip().lower()
    display_name = (profile.get("name") or email or "John").strip()
    if not google_sub or not email:
        flash("Google profile payload was incomplete.", "error")
        return redirect(url_for("login"))
    if email not in settings.allowed_google_email_set:
        clear_auth_session()
        flash("That Google account is not allowed to access this app.", "error")
        return redirect(url_for("login"))

    users = store.t("users")
    user = store.query_one(f"SELECT * FROM {users} WHERE google_sub = %s", (google_sub,))
    if not user:
        user = store.query_one(f"SELECT * FROM {users} WHERE email = %s", (email,))
    if user:
        user_id = int(user["id"])
        store.execute(f"UPDATE {users} SET email = %s, google_sub = %s, display_name = %s WHERE id = %s", (email, google_sub, display_name, user_id))
    else:
        user_id = store.execute(f"INSERT INTO {users} (email, google_sub, display_name) VALUES (%s, %s, %s)", (email, google_sub, display_name))

    session["user_id"] = user_id
    session["email"] = email
    session["display_name"] = display_name
    session["google_sub"] = google_sub
    session["auth_provider"] = "google"
    ensure_user_week(user_id)
    csrf_token()
    return redirect(next_path)


@app.get("/")
@login_required
def index():
    user = session_user()
    assert user is not None
    week_key = ensure_user_week(user["id"])
    schedule_data = get_schedule_data()
    return render_template(
        "schedule.html",
        schedule_data=schedule_data,
        initial_state=build_user_snapshot(user["id"], week_key, schedule_data),
        user=user,
        csrf_token_value=csrf_token(),
        app_timezone=settings.app_timezone,
    )


@app.get("/about")
@login_required
def about():
    user = session_user()
    assert user is not None
    schedule_data = get_schedule_data()
    return render_template(
        "about.html",
        schedule_data=schedule_data,
        user=user,
        app_timezone=settings.app_timezone,
    )


@app.get("/ai-insights")
@login_required
def ai_insights():
    user = session_user()
    assert user is not None
    rows = store.query_all(
        f"SELECT id, week_key, completed_blocks, total_blocks, created_at FROM {store.t('weekly_archive')} WHERE user_id = %s ORDER BY created_at DESC LIMIT 8",
        (user["id"],),
    )
    return render_template(
        "ai_insights.html",
        rows=rows,
        week_label_from_key=week_label_from_key,
        user=user,
        gemini_enabled=settings.gemini_enabled,
    )


@app.get("/ai-insights/<int:archive_id>")
@login_required
def ai_insight_detail(archive_id: int):
    user = session_user()
    assert user is not None
    row = store.query_one(
        f"SELECT id, week_key, snapshot_json, created_at FROM {store.t('weekly_archive')} WHERE id = %s AND user_id = %s",
        (archive_id, user["id"]),
    )
    if not row:
        return "Archive not found.", 404
    snapshot = json.loads(str(row["snapshot_json"]))
    archive_view = archive_view_model(snapshot)
    insight_row = store.query_one(
        f"SELECT analysis_json, model_name, updated_at FROM {store.t('ai_weekly_insights')} WHERE user_id = %s AND archive_id = %s",
        (user["id"], archive_id),
    )
    ai_insight = None
    if insight_row and insight_row.get("analysis_json"):
        try:
            ai_insight = normalize_insight_payload(json.loads(str(insight_row["analysis_json"])))
        except Exception:
            ai_insight = normalize_insight_payload(None, str(insight_row["analysis_json"]))
    return render_template(
        "ai_insight_detail.html",
        archive=row,
        archive_view=archive_view,
        ai_insight=ai_insight,
        ai_model_name=str((insight_row or {}).get("model_name") or ""),
        ai_updated_at=(insight_row or {}).get("updated_at"),
        week_label_from_key=week_label_from_key,
        user=user,
        csrf_token_value=csrf_token(),
        gemini_enabled=settings.gemini_enabled,
    )


@app.get("/history")
@login_required
def history():
    user = session_user()
    assert user is not None
    rows = store.query_all(
        f"SELECT id, week_key, completed_blocks, total_blocks, created_at FROM {store.t('weekly_archive')} WHERE user_id = %s ORDER BY created_at DESC",
        (user["id"],),
    )
    return render_template("history.html", rows=rows, week_label_from_key=week_label_from_key, user=user)


@app.get("/history/compare-last-4")
@login_required
def history_compare_last_four():
    user = session_user()
    assert user is not None
    rows = store.query_all(
        f"SELECT id, week_key, snapshot_json, created_at FROM {store.t('weekly_archive')} WHERE user_id = %s ORDER BY created_at DESC LIMIT 4",
        (user["id"],),
    )
    compare_insight = None
    insight_row = None
    if rows:
        anchor_archive_id = int(rows[0]["id"])
        insight_row = store.query_one(
            f"SELECT analysis_json, model_name, updated_at FROM {store.t('ai_compare_insights')} WHERE user_id = %s AND anchor_archive_id = %s AND week_count = %s",
            (user["id"], anchor_archive_id, len(rows)),
        )
        if insight_row and insight_row.get("analysis_json"):
            try:
                compare_insight = normalize_compare_payload(json.loads(str(insight_row["analysis_json"])))
            except Exception:
                compare_insight = normalize_compare_payload(None, str(insight_row["analysis_json"]))
    return render_template(
        "history_compare.html",
        rows=rows,
        compare_insight=compare_insight,
        ai_model_name=str((insight_row or {}).get("model_name") or ""),
        ai_updated_at=(insight_row or {}).get("updated_at"),
        week_label_from_key=week_label_from_key,
        user=user,
        csrf_token_value=csrf_token(),
        gemini_enabled=settings.gemini_enabled,
    )


@app.post("/history/compare-last-4/analyze-ai")
@login_required
def history_compare_last_four_analyze():
    if not verify_csrf():
        return "Invalid CSRF token.", 400
    user = session_user()
    assert user is not None
    if not settings.gemini_enabled:
        flash("Gemini is not configured yet. Add GEMINI_API_KEY in .env first.", "error")
        return redirect(url_for("history_compare_last_four"))

    rows = store.query_all(
        f"SELECT id, week_key, snapshot_json, created_at FROM {store.t('weekly_archive')} WHERE user_id = %s ORDER BY created_at DESC LIMIT 4",
        (user["id"],),
    )
    if len(rows) < 2:
        flash("You need at least 2 archived weeks before comparison becomes useful.", "error")
        return redirect(url_for("history_compare_last_four"))

    limit_error = check_ai_rate_limit()
    if limit_error:
        flash(limit_error, "error")
        return redirect(url_for("history_compare_last_four"))

    context = build_compare_context(rows)
    try:
        analysis = generate_compare_ai_analysis(context)
    except Exception as exc:
        flash(f"AI comparison failed: {clip_text(exc, 220)}", "error")
        return redirect(url_for("history_compare_last_four"))

    anchor_archive_id = int(rows[0]["id"])
    store.execute(
        f"""
        INSERT INTO {store.t('ai_compare_insights')} (user_id, anchor_archive_id, week_count, model_name, context_json, analysis_json)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            model_name = VALUES(model_name),
            context_json = VALUES(context_json),
            analysis_json = VALUES(analysis_json),
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            user["id"],
            anchor_archive_id,
            len(rows),
            settings.gemini_model,
            json.dumps(context, ensure_ascii=False),
            json.dumps(analysis, ensure_ascii=False),
        ),
    )
    flash("AI comparison across recent weeks updated.", "success")
    return redirect(url_for("history_compare_last_four"))


@app.get("/history/<int:archive_id>")
@login_required
def history_detail(archive_id: int):
    user = session_user()
    assert user is not None
    row = store.query_one(
        f"SELECT id, week_key, snapshot_json, created_at FROM {store.t('weekly_archive')} WHERE id = %s AND user_id = %s",
        (archive_id, user["id"]),
    )
    if not row:
        return "Archive not found.", 404
    snapshot = json.loads(str(row["snapshot_json"]))
    archive_view = archive_view_model(snapshot)
    insight_row = store.query_one(
        f"SELECT analysis_json, model_name, updated_at FROM {store.t('ai_weekly_insights')} WHERE user_id = %s AND archive_id = %s",
        (user["id"], archive_id),
    )
    ai_insight = None
    if insight_row and insight_row.get("analysis_json"):
        try:
            ai_insight = normalize_insight_payload(json.loads(str(insight_row["analysis_json"])))
        except Exception:
            ai_insight = normalize_insight_payload(None, str(insight_row["analysis_json"]))
    return render_template(
        "history_detail.html",
        archive=row,
        archive_view=archive_view,
        ai_insight=ai_insight,
        ai_model_name=str((insight_row or {}).get("model_name") or ""),
        ai_updated_at=(insight_row or {}).get("updated_at"),
        week_label_from_key=week_label_from_key,
        user=user,
        csrf_token_value=csrf_token(),
        gemini_enabled=settings.gemini_enabled,
    )


@app.post("/history/<int:archive_id>/analyze-ai")
@login_required
def history_analyze_ai(archive_id: int):
    if not verify_csrf():
        return "Invalid CSRF token.", 400
    user = session_user()
    assert user is not None
    if not settings.gemini_enabled:
        flash("Gemini is not configured yet. Add GEMINI_API_KEY in .env first.", "error")
        return redirect(url_for("history_detail", archive_id=archive_id))

    limit_error = check_ai_rate_limit()
    if limit_error:
        flash(limit_error, "error")
        return redirect(url_for("history_detail", archive_id=archive_id))

    row = store.query_one(
        f"SELECT id, week_key, snapshot_json FROM {store.t('weekly_archive')} WHERE id = %s AND user_id = %s",
        (archive_id, user["id"]),
    )
    if not row:
        return "Archive not found.", 404

    snapshot = json.loads(str(row["snapshot_json"]))
    archive_view = archive_view_model(snapshot)
    context = build_ai_week_context(row, archive_view)
    try:
        analysis = generate_weekly_ai_analysis(context)
    except Exception as exc:
        flash(f"AI analysis failed: {clip_text(exc, 220)}", "error")
        return redirect(url_for("history_detail", archive_id=archive_id))

    store.execute(
        f"""
        INSERT INTO {store.t('ai_weekly_insights')} (user_id, archive_id, week_key, model_name, context_json, analysis_json)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            week_key = VALUES(week_key),
            model_name = VALUES(model_name),
            context_json = VALUES(context_json),
            analysis_json = VALUES(analysis_json),
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            user["id"],
            archive_id,
            str(row["week_key"] or ""),
            settings.gemini_model,
            json.dumps(context, ensure_ascii=False),
            json.dumps(analysis, ensure_ascii=False),
        ),
    )
    flash("AI weekly analysis updated.", "success")
    return redirect(url_for("history_detail", archive_id=archive_id))


@app.post("/api/blocks/toggle")
@login_required
def api_toggle_block():
    if not verify_csrf():
        return jsonify({"ok": False, "error": "Invalid CSRF token."}), 400
    user = session_user()
    assert user is not None
    week_key = ensure_user_week(user["id"])
    payload = request.get_json(silent=True) or {}
    day_key = str(payload.get("day_key") or "").strip()
    block_idx = payload.get("block_idx")
    completed = bool(payload.get("completed"))
    lookup = _day_lookup()
    if day_key not in lookup or not isinstance(block_idx, int) or block_idx < 0 or block_idx >= len(lookup[day_key]["blocks"]):
        return jsonify({"ok": False, "error": "Invalid block."}), 400
    store.execute(
        f"""
        INSERT INTO {store.t('block_state')} (user_id, week_key, day_key, block_idx, completed)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE completed = VALUES(completed), updated_at = CURRENT_TIMESTAMP
        """,
        (user["id"], week_key, day_key, block_idx, 1 if completed else 0),
    )
    return jsonify({"ok": True})


@app.post("/api/notes")
@login_required
def api_save_note():
    if not verify_csrf():
        return jsonify({"ok": False, "error": "Invalid CSRF token."}), 400
    user = session_user()
    assert user is not None
    week_key = ensure_user_week(user["id"])
    payload = request.get_json(silent=True) or {}
    day_key = str(payload.get("day_key") or "").strip()
    note = str(payload.get("note") or "")
    if day_key not in _day_lookup():
        return jsonify({"ok": False, "error": "Unknown day key."}), 400
    store.execute(
        f"""
        INSERT INTO {store.t('day_notes')} (user_id, week_key, day_key, note_text)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE note_text = VALUES(note_text), updated_at = CURRENT_TIMESTAMP
        """,
        (user["id"], week_key, day_key, note[:12000]),
    )
    return jsonify({"ok": True})


@app.post("/api/preferences")
@login_required
def api_save_preferences():
    if not verify_csrf():
        return jsonify({"ok": False, "error": "Invalid CSRF token."}), 400
    user = session_user()
    assert user is not None
    payload = request.get_json(silent=True) or {}
    store.execute(
        f"""
        INSERT INTO {store.t('user_prefs')} (user_id, compact_mode, hide_completed)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE compact_mode = VALUES(compact_mode), hide_completed = VALUES(hide_completed), updated_at = CURRENT_TIMESTAMP
        """,
        (user["id"], 1 if bool(payload.get("compact_mode")) else 0, 1 if bool(payload.get("hide_completed")) else 0),
    )
    return jsonify({"ok": True})


@app.post("/api/carry-forward")
@login_required
def api_save_carry_forward():
    if not verify_csrf():
        return jsonify({"ok": False, "error": "Invalid CSRF token."}), 400
    user = session_user()
    assert user is not None
    payload = request.get_json(silent=True) or {}
    note = str(payload.get("note") or "")
    store.execute(
        f"""
        INSERT INTO {store.t('carry_forward')} (user_id, note_text)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE note_text = VALUES(note_text), updated_at = CURRENT_TIMESTAMP
        """,
        (user["id"], note[:12000]),
    )
    return jsonify({"ok": True})


@app.post("/api/focus")
@login_required
def api_save_focus():
    if not verify_csrf():
        return jsonify({"ok": False, "error": "Invalid CSRF token."}), 400
    user = session_user()
    assert user is not None
    week_key = ensure_user_week(user["id"])
    payload = request.get_json(silent=True) or {}
    day_key = str(payload.get("day_key") or "").strip()
    block_idx = payload.get("block_idx")
    lookup = _day_lookup()
    if day_key not in lookup or not isinstance(block_idx, int) or block_idx < 0 or block_idx >= len(lookup[day_key]["blocks"]):
        return jsonify({"ok": False, "error": "Invalid focus block."}), 400
    store.execute(
        f"""
        INSERT INTO {store.t('focus_block')} (user_id, week_key, day_key, block_idx)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE block_idx = VALUES(block_idx), updated_at = CURRENT_TIMESTAMP
        """,
        (user["id"], week_key, day_key, block_idx),
    )
    return jsonify({"ok": True})


@app.post("/api/week/reset")
@login_required
def api_reset_week():
    if not verify_csrf():
        return jsonify({"ok": False, "error": "Invalid CSRF token."}), 400
    user = session_user()
    assert user is not None
    week_key = ensure_user_week(user["id"])
    schedule_data = get_schedule_data()
    snapshot = build_user_snapshot(user["id"], week_key, schedule_data)
    has_activity = (
        snapshot["completed_blocks"] > 0
        or any(snapshot["notes"].values())
        or bool(snapshot["focus"])
        or bool(snapshot["carry_forward_note"])
        or review_has_content(snapshot["weekly_review"])
    )

    archive_id = None
    if has_activity:
        archive_id = store.execute(
            f"""
            INSERT INTO {store.t('weekly_archive')} (user_id, week_key, completed_blocks, total_blocks, carry_forward_note, snapshot_json)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (user["id"], week_key, snapshot["completed_blocks"], snapshot["total_blocks"], snapshot["carry_forward_note"], json.dumps(snapshot, ensure_ascii=False)),
        )

    store.execute_many(
        [
            (f"DELETE FROM {store.t('block_state')} WHERE user_id = %s AND week_key = %s", (user["id"], week_key)),
            (f"DELETE FROM {store.t('day_notes')} WHERE user_id = %s AND week_key = %s", (user["id"], week_key)),
            (f"DELETE FROM {store.t('focus_block')} WHERE user_id = %s AND week_key = %s", (user["id"], week_key)),
            (f"DELETE FROM {store.t('weekly_review')} WHERE user_id = %s", (user["id"],)),
        ]
    )
    return jsonify({"ok": True, "archive_id": archive_id, "archived": has_activity})


@app.get("/weekly-review")
@login_required
def weekly_review():
    user = session_user()
    assert user is not None
    review = store.query_one(f"SELECT wins_text, misses_text, adjust_text FROM {store.t('weekly_review')} WHERE user_id = %s", (user["id"],)) or {
        "wins_text": "",
        "misses_text": "",
        "adjust_text": "",
    }
    return render_template("weekly_review.html", review=review, user=user, csrf_token_value=csrf_token())


@app.post("/weekly-review")
@login_required
def weekly_review_save():
    if not verify_csrf():
        return "Invalid CSRF token.", 400
    user = session_user()
    assert user is not None
    store.execute(
        f"""
        INSERT INTO {store.t('weekly_review')} (user_id, wins_text, misses_text, adjust_text)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE wins_text = VALUES(wins_text), misses_text = VALUES(misses_text), adjust_text = VALUES(adjust_text), updated_at = CURRENT_TIMESTAMP
        """,
        (
            user["id"],
            str(request.form.get("wins_text") or "")[:12000],
            str(request.form.get("misses_text") or "")[:12000],
            str(request.form.get("adjust_text") or "")[:12000],
        ),
    )
    flash("Weekly review saved.", "success")
    return redirect(url_for("weekly_review"))


@app.get("/schedule-editor")
@login_required
def schedule_editor():
    user = session_user()
    assert user is not None
    schedule_data = get_schedule_data()
    return render_template("schedule_editor.html", schedule_json=json.dumps(schedule_data, ensure_ascii=False, indent=2), user=user, csrf_token_value=csrf_token())


@app.post("/schedule-editor")
@login_required
def schedule_editor_save():
    if not verify_csrf():
        return "Invalid CSRF token.", 400
    raw_json = str(request.form.get("schedule_json") or "")
    try:
        payload = validate_schedule(json.loads(raw_json))
    except Exception as exc:
        flash(f"Schedule JSON is invalid: {exc}", "error")
        user = session_user()
        assert user is not None
        return render_template("schedule_editor.html", schedule_json=raw_json, user=user, csrf_token_value=csrf_token())
    store.execute(
        f"""
        INSERT INTO {store.t('schedule_config')} (id, schedule_json)
        VALUES (1, %s)
        ON DUPLICATE KEY UPDATE schedule_json = VALUES(schedule_json), updated_at = CURRENT_TIMESTAMP
        """,
        (json.dumps(payload, ensure_ascii=False),),
    )
    flash("Schedule updated.", "success")
    return redirect(url_for("schedule_editor"))


