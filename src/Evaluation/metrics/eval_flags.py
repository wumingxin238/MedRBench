import os

_NO_SEARCH_VALUES = ('1', 'true', 'yes', 'on')


def web_search_enabled() -> bool:
    """Return False when EVAL_DISABLE_WEB_SEARCH is set (CLI: --no-web-search)."""
    return os.environ.get('EVAL_DISABLE_WEB_SEARCH', '0').strip().lower() not in _NO_SEARCH_VALUES


NO_WEB_SEARCH_INFO = (
    '(Web search disabled. Use only the patient case summary and your medical knowledge.)'
)
