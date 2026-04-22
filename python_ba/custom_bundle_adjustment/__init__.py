"""custom_bundle_adjustment — Custom BA 파이프라인.

파일 구조:
    bundle_adjustment.py          공용 라이브러리 (Reconstruction dataclass, load/save, passthrough)
    bundle_adjustment_utils.py    COLMAP binary I/O + 수학 유틸
    methods/{name}.py             BA 방법론. 각 파일은 def {name}(recon, **kw) -> recon 를 노출

run_sba.py 에서:
    --sba-module-name 없음   → bundle_adjustment.passthrough() 사용
    --sba-module-name {name} → methods.{name}.{name}(recon) 호출
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Callable, List

from .bundle_adjustment import Reconstruction, passthrough


def available() -> List[str]:
    """사용 가능한 method {name} 리스트 (methods/ 폴더에서 자동 수집)."""
    mdir = Path(__file__).parent / "methods"
    if not mdir.exists():
        return []
    return sorted(
        info.name for info in pkgutil.iter_modules([str(mdir)])
        if not info.name.startswith("_")
    )


def load(name: str) -> Callable[..., Reconstruction]:
    """methods/{name}.py 에서 함수 {name} 을 꺼내 반환한다."""
    try:
        mod = importlib.import_module(f".methods.{name}", package=__name__)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"methods/{name}.py 를 찾을 수 없습니다. 사용 가능: {available()}"
        ) from None
    fn = getattr(mod, name, None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"methods.{name} 은 동명의 함수 "
            f"`def {name}(recon: Reconstruction, **kwargs) -> Reconstruction` 를 노출해야 합니다."
        )
    return fn


__all__ = ["Reconstruction", "passthrough", "available", "load"]
