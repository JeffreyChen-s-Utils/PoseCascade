"""Phase 11 multi-track timeline editor.

Combines the Phase 3 :class:`TimelineDock` (transport / scrub) with a
tree of editable tracks grouped by PMX display frame. Edit operations
flow through a :class:`CommandStack` so the standard editor flow —
insert keyframe, scrub, undo, redo — falls out of the data layer
without bespoke per-action plumbing.

The widget is deliberately Qt-thin: a :class:`QTreeWidget` is enough
to express "groups of named tracks" plus selection state. Per-track
keyframe strips with bezier-handle dragging are useful but separable
work; they can drop in later without changing the dock's public API.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from vmd.types import VmdBoneKeyframe, VmdMorphKeyframe

from posecascade.animation.commands import (
    CommandStack,
    DeleteBoneKeyframe,
    InsertBoneKeyframe,
    InsertMorphKeyframe,
)
from posecascade.animation.document import AnimationDocument
from posecascade.i18n import t
from posecascade.ui.track_list_model import (
    TrackEntry,
    TrackKind,
    build_track_list,
)

if TYPE_CHECKING:
    from posecascade.assets.types import ImportedScene


_ENTRY_USER_ROLE = 0x0100   # ``Qt.ItemDataRole.UserRole`` numeric value
_DEFAULT_LINEAR_HANDLES = ((20, 20, 107, 107), (20, 20, 107, 107),
                           (20, 20, 107, 107), (20, 20, 107, 107))


class MultiTrackTimelineDock(QDockWidget):
    """Right-dock multi-track editor backed by an :class:`AnimationDocument`."""

    document_changed = Signal()

    def __init__(
        self,
        document: AnimationDocument,
        scene: ImportedScene | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(t("tracks.title"), parent)
        self._document = document
        self._scene = scene
        self._stack = CommandStack()
        self._current_frame = 0
        self._tree: QTreeWidget = QTreeWidget()
        self._frame_label = QLabel(t("tracks.frame_label", value=0))
        self._build_ui()
        self.refresh_tree()

    # ----- public API --------------------------------------------------
    @property
    def document(self) -> AnimationDocument:
        return self._document

    @property
    def stack(self) -> CommandStack:
        return self._stack

    def set_current_frame(self, frame: int) -> None:
        """The bottom transport calls this when the playhead moves."""
        self._current_frame = max(0, int(frame))
        self._frame_label.setText(t("tracks.frame_label", value=self._current_frame))

    def selected_entry(self) -> TrackEntry | None:
        item = self._tree.currentItem()
        if item is None or item.parent() is None:
            return None
        return item.data(0, _ENTRY_USER_ROLE)

    def insert_keyframe_at_current_frame(self) -> bool:
        """Insert a default keyframe on the selected track at the playhead.

        Returns ``True`` when the operation went through to the document
        (caller can use this as a "did anything happen?" hint, e.g. to
        decide whether to flash the Insert button).
        """
        entry = self.selected_entry()
        if entry is None:
            return False
        if entry.kind == TrackKind.BONE:
            self._stack.push(
                InsertBoneKeyframe(
                    document=self._document,
                    keyframe=_default_bone_keyframe(entry.display_name, self._current_frame),
                ),
            )
        elif entry.kind == TrackKind.MORPH:
            self._stack.push(
                InsertMorphKeyframe(
                    document=self._document,
                    keyframe=VmdMorphKeyframe(
                        morph_name=entry.display_name,
                        frame=self._current_frame,
                        weight=0.0,
                    ),
                ),
            )
        else:
            return False
        self._after_edit()
        return True

    def delete_selected_keyframe(self, frame: int | None = None) -> bool:
        """Delete the keyframe at ``frame`` (or the current frame when
        ``None``) on the selected bone track."""
        entry = self.selected_entry()
        if entry is None or entry.kind != TrackKind.BONE:
            return False
        target_frame = self._current_frame if frame is None else int(frame)
        existing = self._document.find_bone_keyframe(entry.display_name, target_frame)
        if existing is None:
            return False
        self._stack.push(
            DeleteBoneKeyframe(
                document=self._document,
                bone_name=entry.display_name,
                frame=target_frame,
            ),
        )
        self._after_edit()
        return True

    def undo(self) -> bool:
        if not self._stack.can_undo():
            return False
        self._stack.undo()
        self._after_edit()
        return True

    def redo(self) -> bool:
        if not self._stack.can_redo():
            return False
        self._stack.redo()
        self._after_edit()
        return True

    def refresh_tree(self) -> None:
        """Rebuild the tree from the current document state.

        Preserves selection across the rebuild by snapshotting the
        selected entry's ``(kind, display_name)`` and re-selecting the
        matching item once the tree's children are repopulated.
        """
        previous = self.selected_entry()
        self._tree.blockSignals(True)
        try:
            self._tree.clear()
            for group in build_track_list(self._document, self._scene):
                group_item = QTreeWidgetItem([group.name, ""])
                for entry in group.entries:
                    child = QTreeWidgetItem(
                        [entry.display_name, str(entry.keyframe_count)],
                    )
                    child.setData(0, _ENTRY_USER_ROLE, entry)
                    group_item.addChild(child)
                    if previous is not None and (
                        entry.kind == previous.kind
                        and entry.display_name == previous.display_name
                    ):
                        self._tree.setCurrentItem(child)
                self._tree.addTopLevelItem(group_item)
                group_item.setExpanded(True)
        finally:
            self._tree.blockSignals(False)

    # ----- internal ----------------------------------------------------
    def _build_ui(self) -> None:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 4, 8, 8)

        controls = QHBoxLayout()
        self._insert_button = QPushButton(t("tracks.action.insert"))
        self._insert_button.clicked.connect(self.insert_keyframe_at_current_frame)
        self._insert_button.setToolTip(t("tracks.tooltip.insert"))
        controls.addWidget(self._insert_button)

        self._delete_button = QPushButton(t("tracks.action.delete"))
        self._delete_button.clicked.connect(lambda: self.delete_selected_keyframe(None))
        self._delete_button.setToolTip(t("tracks.tooltip.delete"))
        controls.addWidget(self._delete_button)

        self._undo_button = QPushButton(t("tracks.action.undo"))
        self._undo_button.clicked.connect(self.undo)
        self._undo_button.setToolTip(t("tracks.tooltip.undo"))
        controls.addWidget(self._undo_button)

        self._redo_button = QPushButton(t("tracks.action.redo"))
        self._redo_button.clicked.connect(self.redo)
        self._redo_button.setToolTip(t("tracks.tooltip.redo"))
        controls.addWidget(self._redo_button)

        controls.addStretch(1)
        controls.addWidget(self._frame_label)
        layout.addLayout(controls)

        self._tree.setHeaderLabels([t("tracks.header.track"), t("tracks.header.keys")])
        self._tree.setColumnCount(2)
        layout.addWidget(self._tree, 1)
        self.setWidget(container)

    def _after_edit(self) -> None:
        self.refresh_tree()
        self.document_changed.emit()


def _default_bone_keyframe(bone_name: str, frame: int) -> VmdBoneKeyframe:
    return VmdBoneKeyframe(
        bone_name=bone_name,
        frame=frame,
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        bezier_handles=_DEFAULT_LINEAR_HANDLES,
    )
