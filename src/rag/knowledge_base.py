"""
Chargement et indexation de la base de connaissances.
"""
import json
from pathlib import Path
from typing import Any


def load_knowledge(base_dir: str | Path = "data/knowledge_base") -> list[dict[str, Any]]:
    """Charge tous les fichiers JSON de la base de connaissances."""
    base_dir = Path(base_dir)
    documents: list[dict[str, Any]] = []
    for json_file in base_dir.glob("*.json"):
        entries = json.loads(json_file.read_text(encoding="utf-8"))
        for entry in entries:
            entry["source"] = json_file.stem
            documents.append(entry)
    return documents


def get_template(category: str, base_dir: str | Path = "data/knowledge_base") -> str | None:
    """Retourne le template de réponse pour une catégorie donnée."""
    path = Path(base_dir) / "response_templates.json"
    templates = json.loads(path.read_text(encoding="utf-8"))
    for tpl in templates:
        if tpl["category"] == category:
            return tpl["template"]
    return None
