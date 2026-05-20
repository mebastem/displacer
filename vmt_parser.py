"""
vmt_parser.py — parse Source Engine VMT material files.
"""
import re
from typing import Dict


def parse_vmt(text: str) -> Dict[str, str]:
    """Return a flat dict of lowercase param key → string value."""
    text = re.sub(r'//[^\n]*', '', text)
    result: Dict[str, str] = {}

    m = re.search(r'"(\w+)"', text)
    if m:
        result['__shader__'] = m.group(1).lower()

    # Strip nested blocks (Proxies etc.) so their contents don't clobber top-level params
    depth = 0
    flat = []
    for ch in text:
        if ch == '{':
            depth += 1
            if depth <= 1:
                flat.append(ch)
        elif ch == '}':
            if depth <= 1:
                flat.append(ch)
            depth = max(0, depth - 1)
        elif depth <= 1:
            flat.append(ch)
    flat_text = ''.join(flat)

    # Match: "key" "value"  OR  "key" bare_value  (top-level only)
    for m in re.finditer(
        r'"?(\$[^"\s]+)"?\s+(?:"([^"]*)"|([^\s"{}\n]+))',
        flat_text, re.IGNORECASE
    ):
        key = m.group(1).lower()
        val = m.group(2) if m.group(2) is not None else m.group(3)
        result[key] = val

    return result


def leaf_no_ext(path: str) -> str:
    """Lowercase filename without directory or extension."""
    name = path.replace('\\', '/').split('/')[-1].lower()
    if '.' in name:
        name = name.rsplit('.', 1)[0]
    return name


def seamless_scale(vmt: Dict[str, str]) -> float:
    try:
        return float(vmt.get('$seamless_scale', 0) or 0)
    except ValueError:
        return 0.0


def is_blend(vmt: Dict[str, str]) -> bool:
    return '$basetexture2' in vmt

