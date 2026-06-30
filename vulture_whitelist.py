# Vulture whitelist for the public sn-79 surface.
#
# At min_confidence=80 (see [tool.vulture] in pyproject.toml) vulture already
# filters out the low-confidence Pydantic-field / abstract-hook false positives
# that dominated the original review (e.g. taos/im/protocol/models.py). Add a
# `_.<name>` reference below only for a genuine framework-invisible use that
# vulture still reports at >=80 confidence (dynamically-dispatched handlers,
# attributes read solely via getattr / serialization, etc.).
#
# Usage: each line `_.<attr>` or `<name>` marks the symbol as used.
