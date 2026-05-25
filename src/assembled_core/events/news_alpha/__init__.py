"""News Alpha — event-driven directional trading on breaking news.

This is the intended Crisis Alpha: detect high-impact news events and
immediately open directional positions in the specific assets affected.

Example: Strait of Hormuz blockade → Long XLE/UCO within the same trading day.

Separate from the defensive basket overlay (events/crisis_alpha), which is
a slow-moving MDD-reduction tool. This module is about alpha generation.
"""
