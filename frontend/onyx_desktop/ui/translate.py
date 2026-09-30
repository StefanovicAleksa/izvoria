# Canvas: desktop_translate_v1
# Path: frontend/onyx_desktop/ui/translate.py

"""
Keeps every on-screen string in the display language, including across a switch at runtime.

A widget's text is set through here rather than with setText(): the call records *how* to produce the text (an
English key, or a function for text built from data), applies it now, and replays it when the language changes.
So a switch re-labels static headings and dynamic lines ("Step 2 of 5 · 3 files") alike, and a view needs no
hand-written refresh code:

    heading = tl.label("Chat")                                  # a new QLabel, kept translated
    tl.text(self._send, "Ask")                                  # an existing widget
    tl.text(self._phase, lambda: tr("Step {n}", n=step))        # text built from data
    tl.tooltip(self._mode, "Verified keeps only ...")
    tl.items(self._mode, ["Standard", "Verified"])              # combo items, by position
    tl.on_retranslate(self, "history", self._render_history)    # anything else: re-render it

Setting a bound widget's text directly with setText() would be overwritten by the next switch, so once a
widget's text is bound, set it through here too (`tl.text(widget, tl.plain(message))` for text that must not be
translated, such as an engine message that already was).

One binding per (widget, slot): binding again replaces it. Bindings of deleted widgets are dropped on the next
replay.
"""

from typing import Callable, Dict, Iterable, Tuple, Union

import shiboken6
from PySide6.QtWidgets import QLabel, QWidget

from onyx_desktop.utils.i18n import tr

Source = Union[str, Callable[[], str]]

_bindings: Dict[Tuple[int, str], Tuple[object, Callable[[str], None], Callable[[], str]]] = {}
# Past this many, a bind first drops the bindings of widgets that no longer exist (chat bubbles, result cards)
_PURGE_AT = 4000


def _producer(source: Source, values: dict) -> Callable[[], str]:
    if callable(source):
        return source
    return lambda: tr(source, **values)


def plain(text: str) -> Callable[[], str]:
    """Text shown as it is, never translated: a user's own words, a file name, an already-translated message."""
    return lambda: text


def bind(owner: object, slot: str, apply: Callable[[str], None], producer: Callable[[], str]) -> None:
    if len(_bindings) > _PURGE_AT:
        _purge()
    _bindings[(id(owner), slot)] = (owner, apply, producer)
    apply(producer())


def text(widget: QWidget, source: Source, **values) -> None:
    bind(widget, "text", widget.setText, _producer(source, values))


def tooltip(widget: QWidget, source: Source, **values) -> None:
    bind(widget, "tooltip", widget.setToolTip, _producer(source, values))


def placeholder(widget: QWidget, source: Source, **values) -> None:
    bind(widget, "placeholder", widget.setPlaceholderText, _producer(source, values))


def title(widget: QWidget, source: Source, **values) -> None:
    bind(widget, "title", widget.setWindowTitle, _producer(source, values))


def items(combo, sources: Iterable[Source]) -> None:
    """Keeps a combo's item texts translated, by position. The items must already exist."""
    for index, source in enumerate(sources):
        bind(combo, f"item{index}", lambda value, i=index: combo.setItemText(i, value), _producer(source, {}))


def on_retranslate(owner: object, slot: str, callback: Callable[[], None]) -> None:
    """Runs `callback` now and after every switch: for content re-rendered from data, like a history list."""
    bind(owner, slot, lambda _unused: callback(), lambda: "")


def label(source: Source, role: str = "", **values) -> QLabel:
    widget = QLabel()
    if role:
        widget.setProperty("role", role)
    text(widget, source, **values)
    return widget


def retranslate_widget(widget: object) -> None:
    """Re-applies one widget's bindings, after the state its producers read has changed."""
    owner_id = id(widget)
    for (bound_id, _slot), (_owner, apply, producer) in list(_bindings.items()):
        if bound_id == owner_id:
            apply(producer())


def retranslate_all() -> None:
    """Replays every binding in the current display language."""
    for key, (owner, apply, producer) in list(_bindings.items()):
        if not _alive(owner):
            _bindings.pop(key, None)
            continue
        apply(producer())


def _alive(owner: object) -> bool:
    try:
        return shiboken6.isValid(owner)
    except TypeError:
        return True   # not a Qt object: a plain Python owner lives as long as the binding holds it


def _purge() -> None:
    for key, (owner, _apply, _producer_) in list(_bindings.items()):
        if not _alive(owner):
            _bindings.pop(key, None)
