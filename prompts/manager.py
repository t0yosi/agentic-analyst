"""Versioned prompt loading.

Prompts live in ``prompts/<component>/v<n>.txt`` so they can be revised without
touching code, and so a past run can be reproduced against the exact prompt
version that produced it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from utils.errors import ConfigError

logger = logging.getLogger(__name__)

PROMPT_ROOT = Path(__file__).parent
_VERSION_FILE = re.compile(r"^v(\d+)\.txt$")


class PromptManager:
    """Resolves prompt text for a component, latest version by default."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self._root = root or PROMPT_ROOT
        self._cache: Dict[str, str] = {}

    def available_versions(self, component: str) -> List[int]:
        """Return every version number on disk for ``component``, ascending."""
        directory = self._root / component
        if not directory.is_dir():
            raise ConfigError(
                f"no prompt directory for component {component!r} at {directory}"
            )
        versions = []
        for path in directory.iterdir():
            match = _VERSION_FILE.match(path.name)
            if match:
                versions.append(int(match.group(1)))
        if not versions:
            raise ConfigError(
                f"prompt directory {directory} contains no vN.txt files"
            )
        return sorted(versions)

    def load(self, component: str, version: Optional[int] = None) -> str:
        """Return prompt text for ``component``.

        Args:
            component: Directory name under ``prompts/``.
            version: Explicit version. Defaults to the highest available.

        Returns:
            The prompt text, stripped of trailing whitespace.

        Raises:
            ConfigError: If the component or the requested version is missing.
        """
        resolved = version if version is not None else self.available_versions(component)[-1]
        key = f"{component}:v{resolved}"
        if key in self._cache:
            return self._cache[key]

        path = self._root / component / f"v{resolved}.txt"
        if not path.is_file():
            raise ConfigError(
                f"prompt not found: {path}; available versions are "
                f"{self.available_versions(component)}"
            )
        text = path.read_text().strip()
        if not text:
            raise ConfigError(f"prompt file {path} is empty")

        self._cache[key] = text
        logger.info("[CONFIG] prompt loaded component=%s version=%d chars=%d",
                    component, resolved, len(text))
        return text


_DEFAULT_MANAGER = PromptManager()


def load_prompt(component: str, version: Optional[int] = None) -> str:
    """Module-level convenience wrapper around a shared :class:`PromptManager`."""
    return _DEFAULT_MANAGER.load(component, version)
