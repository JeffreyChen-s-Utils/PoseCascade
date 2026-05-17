"""Process-wide service registry.

Holds the importer manager, asset cache, and script host so menus and dock
widgets can reach them without constructor injection through every layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from posecascade.animation.cloth_host import ClothHost
from posecascade.animation.foot_planting import FootPlanter
from posecascade.animation.physics_host import PhysicsHost
from posecascade.assets.cache import AssetCache
from posecascade.assets.importer_manager import ImporterManager
from posecascade.assets.types import Mesh, Texture
from posecascade.scripting.host import ScriptHost


@dataclass
class Services:
    """Bundle of long-lived services attached to the running process."""

    importer_manager: ImporterManager
    mesh_cache: AssetCache[Mesh] = field(default_factory=AssetCache)
    texture_cache: AssetCache[Texture] = field(default_factory=AssetCache)
    script_host: ScriptHost = field(default_factory=ScriptHost)
    physics_host: PhysicsHost = field(default_factory=PhysicsHost)
    cloth_host: ClothHost = field(default_factory=ClothHost)
    foot_planter: FootPlanter = field(default_factory=FootPlanter)
    # Latest imported skins / meshes — set by ``attach_scene`` so
    # ``attach_script`` (or other late-binding consumers) can reach
    # them without re-importing. Cleared / replaced on every scene
    # load. The foot planter's auto-sample derivation reads these.
    imported_skins: tuple = field(default_factory=tuple)
    imported_meshes: tuple = field(default_factory=tuple)


def build_services(project_root: Path) -> Services:
    """Construct the default service bundle. Importers are discovered eagerly."""
    importers_root = project_root / "importers"
    importer_manager = ImporterManager(importers_root=importers_root)
    importer_manager.discover()
    services = Services(importer_manager=importer_manager)
    services.physics_host.install_default_forces()
    services.cloth_host.install_default_forces()
    # Share the cloth host's collider list with the spring simulator so
    # body capsules (chest, upper-arm, leg) registered by the declarative
    # runtime push hair / ribbon chains out of the body the same way they
    # push the dress cloth out — without this, hair clips through the
    # dress when the character poses deeply (dog crawl, sitting, etc.).
    services.physics_host.share_colliders_with(services.cloth_host)
    return services
