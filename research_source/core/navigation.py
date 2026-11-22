"""Safe Streamlit navigation across tabs/pages.

Widgets with keys (such as the sidebar radio ``wg_page``) lock their
session-state keys after instantiation in the same Streamlit run. Record
a navigation request on button click, then apply it on the next rerun
*before* creating the sidebar widget.
"""
from __future__ import annotations

PENDING_KEY = "wg_pending_navigation"

def queue_navigation(session_state, page: str, well: str) -> None:
    """Save an intent for the following rerun; never mutate ``wg_page`` here."""
    session_state[PENDING_KEY] = {"page": page, "well": str(well)}

def apply_pending_navigation(session_state, allowed_pages, allowed_wells) -> bool:
    """Apply request before sidebar widget creation, with page/well validation.

    Returns True when a permitted page transition was applied.
    """
    pending = session_state.pop(PENDING_KEY, None)
    if not isinstance(pending, dict):
        return False
    page = pending.get("page")
    well = str(pending.get("well", ""))
    if page not in allowed_pages:
        return False
    session_state["wg_page"] = page
    if well and well in set(map(str, allowed_wells)):
        session_state["chosen_well"] = well
    return True
