import pytest
from tabu_search import TabuSearch, TSP_17

@pytest.mark.parametrize(
    "arr, cost",
    [
        ([1, 4, 13, 7, 8, 6, 17, 14, 15, 3, 11, 10, 2, 5, 9, 12, 16], 2085),
        ([4, 13, 7, 17, 6, 8, 14, 15, 3, 11, 10, 2, 5, 9, 12, 16, 1], 2097)
    ],
)
def test_tsp_cost(arr, cost):
    ts = TabuSearch(num_items=17, data=TSP_17)
    assert ts.tsp(arr) == cost
