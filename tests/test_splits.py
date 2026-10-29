import pytest

from cbce.eval.splits import SPLIT_NAME, split_for_year


@pytest.mark.parametrize("year", [1998, 2002, 2006, 2010, 2014, 2018, 2022])
def test_frozen_test_years(year: int) -> None:
    # Arrange: the years named in the split rule when it was fixed

    # Act
    result = split_for_year(year)

    # Assert
    assert result == "test"


@pytest.mark.parametrize("year", [1996, 1997, 1999, 2000, 2001, 2003, 2008, 2011, 2019, 2020, 2021])
def test_every_other_year_is_dev(year: int) -> None:
    # Arrange / Act
    result = split_for_year(year)

    # Assert
    assert result == "dev"


def test_all_years_covered_by_the_data_are_assigned_and_both_sides_are_nonempty() -> None:
    # Arrange: the corpus spans 1996 to 2022
    years = range(1996, 2023)

    # Act
    assigned = {year: split_for_year(year) for year in years}

    # Assert
    assert sum(side == "test" for side in assigned.values()) == 7
    assert sum(side == "dev" for side in assigned.values()) == 20


def test_split_name_pins_the_rule() -> None:
    # Arrange / Act / Assert: bump the name, and re-run everything, if the rule ever changes
    assert SPLIT_NAME == "years-mod4-eq2-v1"
