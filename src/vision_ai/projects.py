"""프로젝트(작업공간) 관리.

## 왜 나누는가

현장·라인마다 찍히는 물건도, 결함의 모습도, 판정 기준도 다르다. 이것을 한 manifest에
섞으면 두 가지가 동시에 망가진다.

* **모델** — A라인 이미지로 학습한 정상 분포에 B라인 이미지가 섞이면 "정상"의 범위가
  넓어져 결함을 놓친다.
* **판단** — 성능 지표가 여러 현장의 평균이 되어, 어느 라인이 잘 되고 어느 라인이 안 되는지
  알 수 없게 된다.

그래서 데이터·라벨·분할·모델·드리프트 기준선을 프로젝트 단위로 나눈다.

## 무엇을 나누지 않는가

사전학습 CNN 모델(45MB)은 프로젝트와 무관하게 같은 파일이라 공통으로 둔다
(`config.shared_model_dir()`). 프로젝트를 만들 때마다 다시 내려받을 이유가 없다.

## 활성 프로젝트를 파일에 두는 이유

세션에 두면 브라우저 탭마다 다른 프로젝트를 보게 되고, `scripts/`의 측정 스크립트는
어느 프로젝트를 봐야 할지 알 수 없다. 파일에 두면 화면과 스크립트가 같은 것을 본다.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import config

REGISTRY_FILE = "projects.json"
DEFAULT_NAME = "기본 프로젝트"

MAX_NAME = 60
SLUG_FALLBACK = "project"


@dataclass(frozen=True)
class Project:
    slug: str
    name: str
    created_at: str = ""
    note: str = ""


@dataclass
class Registry:
    active: str = config.DEFAULT_PROJECT
    projects: list[Project] = field(default_factory=list)


def registry_path() -> Path:
    """프로젝트 목록 파일. 프로젝트 **바깥**에 있어야 한다."""
    return config.DATA_HOME / REGISTRY_FILE


# 한글 음절을 로마자로 옮기기 위한 자모 표 (문화관광부 로마자 표기법을 단순화한 것).
# 음운 변화(자음 동화 등)는 적용하지 않는다 — 여기서 필요한 것은 «읽어서 알아볼 수 있는
# 폴더 이름»이지 표기법 준수가 아니고, 규칙을 넣을수록 되돌려 읽기가 어려워진다.
_INITIALS = (
    "g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj",
    "ch", "k", "t", "p", "h",
)
_VOWELS = (
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo",
    "u", "wo", "we", "wi", "yu", "eu", "ui", "i",
)
_FINALS = (
    "", "k", "k", "ks", "n", "nj", "nh", "t", "l", "lg", "lm", "lb", "ls", "lt",
    "lp", "lh", "m", "b", "bs", "s", "ss", "ng", "j", "c", "k", "t", "p", "h",
)
_HANGUL_BASE = 0xAC00
_HANGUL_COUNT = 11172

SLUG_HASH_LEN = 4
"""이름이 통째로 지워졌을 때 붙이는 짧은 해시 길이."""


def romanize(text: str) -> str:
    """한글 음절을 로마자로 옮긴다. 한글이 아닌 글자는 그대로 둔다.

    «라인1 검사» → «rain1 geomsa». 완벽한 표기법은 아니지만 **폴더만 보고 어느 현장인지
    알 수 있게** 하는 것이 목적이다.
    """
    out = []
    for char in str(text):
        index = ord(char) - _HANGUL_BASE
        if 0 <= index < _HANGUL_COUNT:
            out.append(_INITIALS[index // 588])
            out.append(_VOWELS[(index % 588) // 28])
            out.append(_FINALS[index % 28])
        else:
            out.append(char)
    return "".join(out)


def slugify(name: str) -> str:
    """폴더 이름으로 쓸 수 있는 slug를 만든다.

    한글 프로젝트명이 기본인데 그대로 폴더명으로 쓰면 OS·인코딩에 따라 문제가 생긴다.
    그렇다고 한글을 지워 버리면 **«라인1 검사»가 통째로 `project`가 되어** 폴더만 봐서는
    어느 현장인지 알 수 없다. 프로젝트가 서너 개만 돼도 `project-2`, `project-3`이 쌓인다.

    그래서 지우지 말고 **로마자로 옮긴다** — `rain1-geomsa`. 그래도 남는 것이 없으면
    (한자·이모지만 있는 이름) 이름의 짧은 해시를 붙여 서로 구분되게 한다.
    """
    label = str(name).strip()
    text = unicodedata.normalize("NFKD", romanize(label)).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:40].strip("-")
    if text:
        return text
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:SLUG_HASH_LEN]
    return f"{SLUG_FALLBACK}-{digest}" if label else SLUG_FALLBACK


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_registry() -> Registry:
    return Registry(
        active=config.DEFAULT_PROJECT,
        projects=[Project(config.DEFAULT_PROJECT, DEFAULT_NAME, _now())],
    )


def load() -> Registry:
    """프로젝트 목록을 읽는다. 없거나 깨졌으면 기본 프로젝트 하나만 있는 것으로 본다."""
    path = registry_path()
    if not path.exists():
        return _default_registry()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        projects = [Project(**item) for item in payload.get("projects", [])]
    except (OSError, ValueError, TypeError):
        return _default_registry()
    if not projects:
        return _default_registry()

    active = str(payload.get("active") or projects[0].slug)
    if active not in {p.slug for p in projects}:
        active = projects[0].slug
    return Registry(active=active, projects=projects)


def save(registry: Registry) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "active": registry.active,
        "projects": [asdict(p) for p in registry.projects],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def bootstrap() -> Registry:
    """앱·스크립트 시작 시 한 번 부른다. 저장된 활성 프로젝트를 실제로 적용한다."""
    registry = load()
    config.use_project(registry.active)
    config.ensure_dirs()
    return registry


def listed() -> list[Project]:
    return load().projects


def get(slug: str) -> Project | None:
    return next((p for p in load().projects if p.slug == slug), None)


def active() -> Project:
    registry = load()
    found = next((p for p in registry.projects if p.slug == registry.active), None)
    return found or registry.projects[0]


def use(slug: str) -> None:
    """활성 프로젝트를 바꾸고 파일에 남긴다."""
    registry = load()
    if slug not in {p.slug for p in registry.projects}:
        raise ValueError(f"없는 프로젝트입니다: {slug}")
    registry.active = slug
    save(registry)
    config.use_project(slug)
    config.ensure_dirs()


def create(name: str, *, note: str = "", activate: bool = True) -> Project:
    """새 프로젝트를 만든다. 이름이 겹쳐도 slug는 겹치지 않게 번호를 붙인다."""
    label = str(name).strip()[:MAX_NAME]
    if not label:
        raise ValueError("프로젝트 이름이 비어 있습니다.")

    registry = load()
    taken = {p.slug for p in registry.projects}
    base = slugify(label)
    slug, index = base, 2
    while slug in taken:
        slug, index = f"{base}-{index}", index + 1

    project = Project(slug=slug, name=label, created_at=_now(), note=str(note))
    registry.projects.append(project)
    if activate:
        registry.active = slug
    save(registry)
    if activate:
        config.use_project(slug)
    config.ensure_dirs()
    return project


def rename(slug: str, name: str) -> Project:
    """이름만 바꾼다. **slug(폴더)는 그대로 둔다** — 바꾸면 쌓인 데이터를 옮겨야 한다."""
    label = str(name).strip()[:MAX_NAME]
    if not label:
        raise ValueError("프로젝트 이름이 비어 있습니다.")
    registry = load()
    updated: Project | None = None
    for index, project in enumerate(registry.projects):
        if project.slug == slug:
            updated = Project(project.slug, label, project.created_at, project.note)
            registry.projects[index] = updated
    if updated is None:
        raise ValueError(f"없는 프로젝트입니다: {slug}")
    save(registry)
    return updated


def remove(slug: str) -> None:
    """목록에서 뺀다. **파일은 지우지 않는다.**

    실수로 지웠을 때 라벨링에 들인 시간이 통째로 사라지면 안 된다. 목록에서만 빼고
    폴더는 남겨 두므로, 잘못 지웠으면 같은 이름으로 다시 만들면 그대로 돌아온다.
    """
    registry = load()
    if len(registry.projects) <= 1:
        raise ValueError("마지막 프로젝트는 지울 수 없습니다.")
    if slug not in {p.slug for p in registry.projects}:
        raise ValueError(f"없는 프로젝트입니다: {slug}")

    registry.projects = [p for p in registry.projects if p.slug != slug]
    if registry.active == slug:
        registry.active = registry.projects[0].slug
        config.use_project(registry.active)
    save(registry)
    config.ensure_dirs()


# --- 예전 배치에서 옮겨오기 ------------------------------------------------
#
# 프로젝트 개념이 생기기 전에는 data/ 와 artifacts/ 바로 아래에 파일이 있었다. 그 배치를
# "기본 프로젝트"로 남겨 두면 코드 곳곳에 "기본 프로젝트만 예외" 분기가 영구히 남는다.
# 한 번 옮기고 끝내는 편이 낫다.
#
# 옮기지 않는 것: projects/ 자신, 프로젝트 목록 파일, 공통 모델 보관소.

_KEEP_IN_PLACE = frozenset({config.PROJECTS_DIR, REGISTRY_FILE, "shared"})


def _legacy_entries(home: Path) -> list[Path]:
    if not home.exists():
        return []
    return sorted(p for p in home.iterdir() if p.name not in _KEEP_IN_PLACE)


def _move_into(source: Path, target: Path, moved: list[str], skipped: list[str]) -> None:
    """`source`를 `target`으로 옮긴다.

    **둘 다 디렉터리면 안쪽 항목별로 합친다.** 대상 디렉터리가 있다는 이유만으로 통째로
    건너뛰면 안 된다 — 앱이 시작할 때 `ensure_dirs()`가 빈 디렉터리를 미리 만들어 두므로,
    그것을 충돌로 보면 정작 안에 든 이미지가 옛 자리에 남는다.

    파일 대 파일 충돌만 건너뛴다. 덮어쓰면 되돌릴 수 없다.
    """
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        moved.append(f"{source} → {target}")
        return

    if source.is_dir() and target.is_dir():
        for child in sorted(source.iterdir()):
            _move_into(child, target / child.name, moved, skipped)
        try:
            source.rmdir()   # 다 옮겼으면 빈 껍데기를 치운다
        except OSError:
            pass
        return

    skipped.append(str(source))


def migration_plan(slug: str = config.DEFAULT_PROJECT) -> dict:
    """무엇을 옮기게 되는지 미리 보여준다. 아무것도 건드리지 않는다."""
    plans = []
    for home in (config.DATA_HOME, config.ARTIFACT_HOME):
        entries = _legacy_entries(home)
        if entries:
            plans.append(
                {
                    "from": home,
                    "to": home / config.PROJECTS_DIR / slug,
                    "entries": [p.name for p in entries],
                }
            )
    return {"slug": slug, "moves": plans, "needed": bool(plans)}


def migrate_legacy(slug: str = config.DEFAULT_PROJECT) -> dict:
    """예전 배치의 파일을 프로젝트 폴더로 옮긴다.

    같은 파일시스템 안의 이름 바꾸기라 크기와 무관하게 즉시 끝난다(VisA 1.8GB도 마찬가지).
    이미 대상에 같은 이름이 있으면 **건너뛴다** — 덮어쓰면 되돌릴 수 없다.

    사전학습 CNN 모델은 프로젝트 안이 아니라 공통 보관소로 보낸다.
    """
    moved: list[str] = []
    skipped: list[str] = []

    # 공통 모델 먼저 — 프로젝트로 딸려 들어가면 안 된다.
    legacy_onnx = config.ARTIFACT_HOME / "models" / "onnx"
    if legacy_onnx.is_dir():
        _move_into(legacy_onnx, config.shared_model_dir() / "onnx", moved, skipped)

    for home in (config.DATA_HOME, config.ARTIFACT_HOME):
        destination = home / config.PROJECTS_DIR / slug
        entries = _legacy_entries(home)
        if not entries:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            _move_into(entry, destination / entry.name, moved, skipped)

    registry = load()
    if slug not in {p.slug for p in registry.projects}:
        registry.projects.append(Project(slug, DEFAULT_NAME, _now()))
    registry.active = slug
    save(registry)
    config.use_project(slug)
    config.ensure_dirs()
    rewritten = _relativize_registry()
    return {"moved": moved, "skipped": skipped, "rewritten": rewritten}


def _relativize_registry() -> int:
    """모델 레지스트리에 남은 **절대경로를 상대경로로** 고쳐 쓴다.

    폴더를 옮기면 절대경로는 그대로 깨진다 — 승격된 모델을 못 찾아 배치 추론과 롤백이
    동시에 멈춘다. 옮긴 직후에 바로잡아야 한다. 옮기기 전 경로에서 파일 이름 쪽 꼬리만
    떼어 새 위치에 붙인다.
    """
    from . import registry as model_registry

    path = model_registry._registry_path()
    if not path.exists():
        return 0
    try:
        import pandas as pd

        frame = pd.read_csv(path)
    except (OSError, ValueError):
        return 0
    if "artifact" not in frame.columns:
        return 0

    root = config.artifact_root()
    changed = 0

    def fix(value) -> object:
        nonlocal changed
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            return value
        parts = Path(value).parts
        if model_registry.REGISTRY_DIR not in parts:
            return value
        tail = Path(*parts[parts.index(model_registry.REGISTRY_DIR):])
        if not (root / tail).exists():
            return value
        changed += 1
        return str(tail)

    frame["artifact"] = frame["artifact"].map(fix)
    if changed:
        frame.to_csv(path, index=False)
    return changed


def summary(slug: str) -> dict:
    """프로젝트 하나의 규모. 목록 화면에서 어느 것이 비어 있는지 알려면 필요하다."""
    data = config.DATA_HOME / config.PROJECTS_DIR / slug
    manifest = data / "manifest.csv"
    images = 0
    if manifest.exists():
        try:
            import pandas as pd

            images = len(pd.read_csv(manifest, usecols=["image_id"]))
        except (OSError, ValueError, KeyError):
            images = 0
    return {"slug": slug, "images": images, "exists": data.exists()}
