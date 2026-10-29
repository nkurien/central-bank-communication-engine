"""Dev/test split of the labelled FOMC sentences. The rule is frozen.

Whole calendar years are assigned to test, so no meeting (or year) has sentences on both
sides. The rule was fixed from the years alone, before any label was inspected. Prompts and
supervised models are tuned on dev only; the test split is read once, after a prompt version
is frozen. Changing this rule invalidates every earlier result, so it needs a new split name.
"""

from typing import Literal

SPLIT_NAME = "years-mod4-eq2-v1"
Split = Literal["dev", "test"]


def split_for_year(year: int) -> Split:
    """Test years are 1998, 2002, 2006, 2010, 2014, 2018 and 2022."""
    return "test" if year % 4 == 2 else "dev"
