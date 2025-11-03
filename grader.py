#!/usr/bin/python3

import json
from typing import List, Optional
from math import radians

import graderUtil
import util
from mapUtil import (
    CityMap,
    checkValid,
    createGridMap,
    createGridMapWithCustomTags,
    createStanfordMap,
    getTotalCost,
    locationFromTag,
    makeGridLabel,
    makeTag,
    RADIUS_EARTH,
)

grader = graderUtil.Grader()
submission = grader.load("submission")

############################################################
# check python version

import sys
import warnings

if not (sys.version_info[0]==3 and sys.version_info[1]==12):
    warnings.warn("Must be using Python 3.12 \n")


def extractPath(startLocation: str, search: util.SearchAlgorithm) -> List[str]:
    """
    Assumes that `solve()` has already been called on the `searchAlgorithm`.

    We extract a sequence of locations from `search.path` (see util.py to better
    understand exactly how this list gets populated).
    """
    return [startLocation] + search.actions


def printPath(
    path: List[str],
    waypointTags: List[str],
    cityMap: CityMap,
    outPath: Optional[str] = "path.json",
):
    doneWaypointTags = set()
    for location in path:
        for tag in cityMap.tags[location]:
            if tag in waypointTags:
                doneWaypointTags.add(tag)
        tagsStr = " ".join(cityMap.tags[location])
        doneTagsStr = " ".join(sorted(doneWaypointTags))
        print(f"Location {location} tags:[{tagsStr}]; done:[{doneTagsStr}]")
    print(f"Total distance: {getTotalCost(path, cityMap)}")

    # (Optional) Write path to file, for use with `visualize.py`
    if outPath is not None:
        with open(outPath, "w") as f:
            data = {"waypointTags": waypointTags, "path": path}
            json.dump(data, f, indent=2)


# Instantiate the Stanford Map as a constant --> just load once!
stanfordMap = createStanfordMap()

########################################################################################
# Problem 1: Grid City

grader.add_manual_part("1a", max_points=2, description="minimum cost path")
grader.add_manual_part("1b", max_points=3, description="UCS basic behavior")
grader.add_manual_part("1c", max_points=3, description="UCS search behavior")

########################################################################################
# Problem 2a: Modeling the Shortest Path Problem.


def t_2a(
    cityMap: CityMap,
    startLocation: str,
    endTag: str,
    expectedCost: Optional[float] = None,
):
    """
    Run UCS on a ShortestPathProblem, specified by
        (startLocation, endTag).
    Check that the cost of the minimum cost path is `expectedCost`.
    """
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(submission.ShortestPathProblem(startLocation, endTag, cityMap))
    path = extractPath(startLocation, ucs)
    grader.require_is_true(checkValid(path, cityMap, startLocation, endTag, []))
    if expectedCost is not None:
        grader.require_is_equal(expectedCost, getTotalCost(path, cityMap))



grader.add_basic_part(
    "2a-1-basic",
    lambda: t_2a(
        cityMap=createGridMap(3, 5),
        startLocation=makeGridLabel(0, 0),
        endTag=makeTag("label", makeGridLabel(2, 2)),
        expectedCost=4,
    ),
    max_points=0.5,
    max_seconds=1,
    description="shortest path on small grid",
)

grader.add_basic_part(
    "2a-2-basic",
    lambda: t_2a(
        cityMap=createGridMap(30, 30),
        startLocation=makeGridLabel(20, 10),
        endTag=makeTag("x", "5"),
        expectedCost=15,
    ),
    max_points=0.5,
    max_seconds=1,
    description="shortest path with multiple end locations",
)

grader.add_hidden_part(
    "2a-3-hidden",
    lambda: t_2a(
        cityMap=createGridMap(100, 100),
        startLocation=makeGridLabel(0, 0),
        endTag=makeTag("label", makeGridLabel(99, 99)),
    ),
    max_points=0.5,
    max_seconds=1,
    description="shortest path with larger grid",
)

# Problem 2a (continued): full Stanford map...
grader.add_basic_part(
    "2a-4-basic",
    lambda: t_2a(
        cityMap=stanfordMap,
        startLocation=locationFromTag(makeTag("landmark", "gates"), stanfordMap),
        endTag=makeTag("landmark", "oval"),
        expectedCost=446.99724421432353,
    ),
    max_points=0.5,
    max_seconds=1,
    description="basic shortest path test case (2a-4)",
)

grader.add_basic_part(
    "2a-5-basic",
    lambda: t_2a(
        cityMap=stanfordMap,
        startLocation=locationFromTag(makeTag("landmark", "rains"), stanfordMap),
        endTag=makeTag("landmark", "evgr_a"),
        expectedCost=660.9598696201658,
    ),
    max_points=0.5,
    max_seconds=1,
    description="basic shortest path test case (2a-5)",
)

grader.add_basic_part(
    "2a-6-basic",
    lambda: t_2a(
        cityMap=stanfordMap,
        startLocation=locationFromTag(makeTag("landmark", "rains"), stanfordMap),
        endTag=makeTag("landmark", "bookstore"),
        expectedCost=1109.3271626156256,
    ),
    max_points=0.5,
    max_seconds=1,
    description="basic shortest path test case (2a-6)",
)

grader.add_hidden_part(
    "2a-7-hidden",
    lambda: t_2a(
        cityMap=stanfordMap,
        startLocation=locationFromTag(makeTag("landmark", "hoover_tower"), stanfordMap),
        endTag=makeTag("landmark", "cantor_arts_center"),
    ),
    max_points=0.5,
    max_seconds=1,
    description="hidden shortest path test case (2a-7)",
)

grader.add_hidden_part(
    "2a-8-hidden",
    lambda: t_2a(
        cityMap=stanfordMap,
        startLocation=locationFromTag(makeTag("landmark", "rains"), stanfordMap),
        endTag=makeTag("landmark", "gates"),
    ),
    max_points=0.5,
    max_seconds=1,
    description="hidden shortest path test case (2a-8)",
)

########################################################################################
# Problem 2b: Custom -- Plan a Route through Stanford


def t_2b_custom():
    """Given custom ShortestPathProblem, output path for visualization."""
    problem = submission.getStanfordShortestPathProblem()
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(problem)
    path = extractPath(problem.startLocation, ucs)
    printPath(path=path, waypointTags=[], cityMap=stanfordMap)
    grader.require_is_true(
        checkValid(path, stanfordMap, problem.startLocation, problem.endTag, [])
    )


grader.add_basic_part(
    "2b-custom",
    t_2b_custom,
    max_points=0,
    max_seconds=10,
    description="customized shortest path through Stanford",
)
grader.add_manual_part("2b", max_points=3, description="customized shortest path through Stanford")


########################################################################################
# Problem 2c: Externalities
grader.add_manual_part("2c", max_points=3, description="externalities of algorithm")


########################################################################################
# Problem 3a: Modeling the Waypoints Shortest Path Problem.


def t_3ab(
    cityMap: CityMap,
    startLocation: str,
    endTag: str,
    waypointTags: List[str],
    expectedCost: Optional[float] = None,
):
    """
    Run UCS on a WaypointsShortestPathProblem, specified by
        (startLocation, waypointTags, endTag).
    """
    ucs = util.UniformCostSearch(verbose=0)
    problem = submission.WaypointsShortestPathProblem(
        startLocation,
        waypointTags,
        endTag,
        cityMap,
    )
    ucs.solve(problem)
    grader.require_is_true(ucs.pathCost is not None)
    path = extractPath(startLocation, ucs)
    grader.require_is_true(
        checkValid(path, cityMap, startLocation, endTag, waypointTags)
    )
    if expectedCost is not None:
        grader.require_is_equal(expectedCost, getTotalCost(path, cityMap))



grader.add_basic_part(
    "3a-1-basic",
    lambda: t_3ab(
        cityMap=createGridMap(3, 5),
        startLocation=makeGridLabel(0, 0),
        waypointTags=[makeTag("y", 4)],
        endTag=makeTag("label", makeGridLabel(2, 2)),
        expectedCost=8,
    ),
    max_points=0.5,
    max_seconds=3,
    description="shortest path on small grid with 1 waypoint",
)

grader.add_basic_part(
    "3a-2-basic",
    lambda: t_3ab(
        cityMap=createGridMap(30, 30),
        startLocation=makeGridLabel(20, 10),
        waypointTags=[makeTag("x", 5), makeTag("x", 7)],
        endTag=makeTag("label", makeGridLabel(3, 3)),
        expectedCost=24.0,
    ),
    max_points=0.5,
    max_seconds=3,
    description="shortest path on medium grid with 2 waypoints",
)

grader.add_basic_part(
    "3a-3-basic",
    lambda: t_3ab(
        cityMap=createGridMapWithCustomTags(2, 2, {(0,0): [], (0,1): ["food", "fuel", "books"], (1,0): ["food"], (1,1): ["fuel"]}),
        startLocation=makeGridLabel(0, 0),
        waypointTags=[
            "food", "fuel", "books"
        ],
        endTag=makeTag("label", makeGridLabel(0, 1)),
        expectedCost=1.0,
    ),
    max_points=0.5,
    max_seconds=3,
    description="shortest path with 3 waypoints and some locations covering multiple waypoints",
)

grader.add_basic_part(
    "3a-4-basic",
    lambda: t_3ab(
        cityMap=createGridMapWithCustomTags(2, 2, {(0,0): ["food"], (0,1): ["fuel"], (1,0): ["food"], (1,1): ["food", "fuel"]}),
        startLocation=makeGridLabel(0, 0),
        waypointTags=[
            "food", "fuel"
        ],
        endTag=makeTag("label", makeGridLabel(0, 1)),
        expectedCost=1.0,
    ),
    max_points=0.5,
    max_seconds=3,
    description="shortest path with 3 waypoints and start location covering some waypoints",
)

grader.add_hidden_part(
    "3a-5-hidden",
    lambda: t_3ab(
        cityMap=createGridMap(100, 100),
        startLocation=makeGridLabel(0, 0),
        waypointTags=[
            makeTag("x", 90),
            makeTag("x", 95),
            makeTag("label", makeGridLabel(3, 99)),
            makeTag("label", makeGridLabel(99, 3)),
        ],
        endTag=makeTag("y", 95),
    ),
    max_points=1,
    max_seconds=3,
    description="shortest path with 4 waypoints and multiple end locations",
)

# Problem 3a (continued): full Stanford map...
grader.add_basic_part(
    "3a-6-basic",
    lambda: t_3ab(
        cityMap=stanfordMap,
        startLocation=locationFromTag(makeTag("landmark", "gates"), stanfordMap),
        waypointTags=[makeTag("landmark", "hoover_tower")],
        endTag=makeTag("landmark", "oval"),
        expectedCost=1108.3623108845995,
    ),
    max_points=0.5,
    max_seconds=3,
    description="basic waypoints test case (3a-4)",
)

grader.add_basic_part(
    "3a-7-basic",
    lambda: t_3ab(
        cityMap=stanfordMap,
        startLocation=locationFromTag(makeTag("landmark", "evgr_a"), stanfordMap),
        waypointTags=[
            makeTag("landmark", "memorial_church"),
            makeTag("landmark", "tresidder"),
            makeTag("landmark", "gates"),
        ],
        endTag=makeTag("landmark", "evgr_a"),
        expectedCost=3381.952714299139,
    ),
    max_points=0.5,
    max_seconds=3,
    description="basic waypoints test case (3a-5)",
)

grader.add_basic_part(
    "3a-8-basic",
    lambda: t_3ab(
        cityMap=stanfordMap,
        startLocation=locationFromTag(makeTag("landmark", "rains"), stanfordMap),
        waypointTags=[
            makeTag("landmark", "gates"),
            makeTag("landmark", "AOERC"),
            makeTag("landmark", "bookstore"),
            makeTag("landmark", "hoover_tower"),
        ],
        endTag=makeTag("landmark", "green_library"),
        expectedCost=3946.478546309725,
    ),
    max_points=1,
    max_seconds=3,
    description="basic waypoints test case (3a-6)",
)

grader.add_hidden_part(
    "3a-9-hidden",
    lambda: t_3ab(
        cityMap=stanfordMap,
        startLocation=locationFromTag(makeTag("landmark", "oval"), stanfordMap),
        waypointTags=[
            makeTag("landmark", "memorial_church"),
            makeTag("landmark", "hoover_tower"),
            makeTag("landmark", "bookstore"),
        ],
        endTag=makeTag("landmark", "AOERC"),
    ),
    max_points=0.5,
    max_seconds=3,
    description="hidden waypoints test case (3a-7)",
)

grader.add_hidden_part(
    "3a-10-hidden",
    lambda: t_3ab(
        cityMap=stanfordMap,
        startLocation=locationFromTag(makeTag("landmark", "oval"), stanfordMap),
        waypointTags=[
            makeTag("landmark", "memorial_church"),
            makeTag("landmark", "stanford_stadium"),
            makeTag("landmark", "rains"),
        ],
        endTag=makeTag("landmark", "oval"),
    ),
    max_points=0.5,
    max_seconds=3,
    description="hidden waypoints test case (3a-8)",
)

grader.add_hidden_part(
    "3a-11-hidden",
    lambda: t_3ab(
        cityMap=stanfordMap,
        startLocation=locationFromTag(makeTag("landmark", "gates"), stanfordMap),
        waypointTags=[
            makeTag("landmark", "lathrop_library"),
            makeTag("landmark", "green_library"),
            makeTag("landmark", "tresidder"),
            makeTag("landmark", "AOERC"),
        ],
        endTag=makeTag("landmark", "evgr_a"),
    ),
    max_points=1,
    max_seconds=5,
    description="hidden waypoints test case (3a-9)",
)

########################################################################################
# Problem 3b: Maximum states with waypoints
grader.add_manual_part("3b", max_points=2, description="max states with waypoints")


########################################################################################
# Problem 3c: Custom -- Plan a Route with Unordered Waypoints through Stanford


def t_3c_custom():
    """Given custom WaypointsShortestPathProblem, output path for visualization."""
    problem = submission.getStanfordWaypointsShortestPathProblem()
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(problem)
    path = extractPath(problem.startLocation, ucs)
    printPath(path=path, waypointTags=problem.waypointTags, cityMap=stanfordMap)
    grader.require_is_true(
        checkValid(
            path,
            stanfordMap,
            problem.startLocation,
            problem.endTag,
            problem.waypointTags,
        )
    )


grader.add_basic_part(
    "3c-custom",
    t_3c_custom,
    max_points=0,
    max_seconds=10,
    description="customized shortest path with waypoints through Stanford",
)
grader.add_manual_part("3c", max_points=3, description="customized shortest path with waypoints through Stanford")

########################################################################################
# Problem 3d: Ethical Considerations
grader.add_manual_part("3d", max_points=3, description="ethical considerations")


########################################################################################
# Problem 4a: A* to UCS reduction

class ZeroHeuristic(util.Heuristic):
    """Estimates the cost between locations as 0 distance."""
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

    def evaluate(self, state: util.State) -> float:
        return 0.0

# Calculates distance only along the north south direction
class NorthSouthHeuristic(util.Heuristic):
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap
        self.endGeoLocations = [
            self.cityMap.geoLocations[location]
            for location, tags in self.cityMap.tags.items()
            if endTag in tags
        ]

    def evaluate(self, state: util.State) -> float:
        currentGeoLocation = self.cityMap.geoLocations[state.location]
        return min(
            RADIUS_EARTH * radians(abs(endGeoLocation.latitude - currentGeoLocation.latitude))
            for endGeoLocation in self.endGeoLocations
        )

def t_4a(
    cityMap: CityMap,
    startLocation: str,
    endTag: str,
    expectedCost: Optional[float] = None,
    heuristic_cls: Optional[type[util.Heuristic]] = ZeroHeuristic,
):
    """
    Run UCS on the A* Reduction of a ShortestPathProblem, specified by
        (startLocation, endTag).
    """
    heuristic = heuristic_cls(endTag, cityMap)

    # Define the baseProblem and corresponding reduction (using `zeroHeuristic`)
    baseProblem = submission.ShortestPathProblem(startLocation, endTag, cityMap)
    aStarProblem = submission.aStarReduction(baseProblem, heuristic)

    # Solve the reduction via a call to `ucs.solve` (similar to prior tests)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(aStarProblem)
    path = extractPath(startLocation, ucs)
    grader.require_is_true(checkValid(path, cityMap, startLocation, endTag, []))
    if expectedCost is not None:
        grader.require_is_equal(expectedCost, getTotalCost(path, cityMap), tolerance=1e-2)



grader.add_basic_part(
    "4a-1-basic",
    lambda: t_4a(
        cityMap=createGridMap(3, 5),
        startLocation=makeGridLabel(0, 0),
        endTag=makeTag("label", makeGridLabel(2, 2)),
        expectedCost=4,
    ),
    max_points=1,
    max_seconds=1,
    description="A* shortest path on small grid",
)

grader.add_basic_part(
    "4a-2-basic",
    lambda: t_4a(
        cityMap=createGridMap(30, 30),
        startLocation=makeGridLabel(20, 10),
        endTag=makeTag("x", "5"),
        expectedCost=15,
    ),
    max_points=1,
    max_seconds=1,
    description="A* shortest path with multiple end locations",
)

grader.add_basic_part(
    "4a-3-basic",
    lambda: t_4a(
        cityMap=stanfordMap,
        startLocation=locationFromTag(makeTag("landmark", "rains"), stanfordMap),
        endTag = makeTag("landmark", "lathrop_library"),
        heuristic_cls=NorthSouthHeuristic,
        expectedCost=1586.1140252970856,
    ),
    max_points=1,
    max_seconds=2,
    description="A* with nontrivial heuristic on Stanford map",
)

grader.add_hidden_part(
    "4a-4-hidden",
    lambda: t_4a(
        cityMap=createGridMap(100, 100),
        startLocation=makeGridLabel(0, 0),
        endTag=makeTag("label", makeGridLabel(99, 99)),
    ),
    max_points=2,
    max_seconds=2,
    description="A* shortest path with larger grid",
)


########################################################################################
# Problem 4b: "straight-line" heuristic for A*


def t_4b_heuristic(
    cityMap: CityMap,
    startLocation: str,
    endTag: str,
    expectedCost: Optional[float] = None,
):
    """Targeted test for `StraightLineHeuristic` to ensure correctness."""
    heuristic = submission.StraightLineHeuristic(endTag, cityMap)
    heuristicCost = heuristic.evaluate(util.State(startLocation))
    if expectedCost is not None:
        grader.require_is_equal(expectedCost, heuristicCost)



grader.add_basic_part(
    "4b-heuristic-1-basic",
    lambda: t_4b_heuristic(
        cityMap=createGridMap(3, 5),
        startLocation=makeGridLabel(0, 0),
        endTag=makeTag("label", makeGridLabel(2, 2)),
        expectedCost=3.145067466556296,
    ),
    max_points=0.5,
    max_seconds=1,
    description="basic straight line heuristic unit test",
)

grader.add_hidden_part(
    "4b-heuristic-2-hidden",
    lambda: t_4b_heuristic(
        cityMap=createGridMap(100, 100),
        startLocation=makeGridLabel(0, 0),
        endTag=makeTag("label", makeGridLabel(99, 99)),
    ),
    max_points=0.5,
    max_seconds=1,
    description="hidden straight line heuristic unit test",
)


# Initialize a `StraightLineHeuristic` using `endTag4b` and the `stanfordMap`
endTag4b = makeTag("landmark", "gates")
if grader.selectedPartName in [
    "4b-astar-1-basic",
    "4b-astar-2-basic",
    "4b-astar-3-hidden",
    "4b-astar-4-hidden",
    None,
]:
    try:
        stanfordStraightLineHeuristic = submission.StraightLineHeuristic(
            endTag4b, stanfordMap
        )
    except:
        stanfordStraightLineHeuristic = None

def t_4b_aStar(
    startLocation: str, heuristic: util.Heuristic, expectedCost: Optional[float] = None
):
    """Run UCS on the A* Reduction of a ShortestPathProblem, w/ `heuristic`"""
    baseProblem = submission.ShortestPathProblem(startLocation, endTag4b, stanfordMap)
    aStarProblem = submission.aStarReduction(baseProblem, heuristic)

    # Solve the reduction via a call to `ucs.solve` (similar to prior tests)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(aStarProblem)
    path = extractPath(startLocation, ucs)
    grader.require_is_true(checkValid(path, stanfordMap, startLocation, endTag4b, []))
    if expectedCost is not None:
        grader.require_is_equal(expectedCost, getTotalCost(path, stanfordMap))



grader.add_basic_part(
    "4b-astar-1-basic",
    lambda: t_4b_aStar(
        startLocation=locationFromTag(makeTag("landmark", "oval"), stanfordMap),
        heuristic=stanfordStraightLineHeuristic,
        expectedCost=446.9972442143235,
    ),
    max_points=0.5,
    max_seconds=2,
    description="basic straight line heuristic A* on Stanford map (4b-astar-1)",
)


grader.add_basic_part(
    "4b-astar-2-basic",
    lambda: t_4b_aStar(
        startLocation=locationFromTag(makeTag("landmark", "rains"), stanfordMap),
        heuristic=stanfordStraightLineHeuristic,
        expectedCost=2005.4388573303765,
    ),
    max_points=1,
    max_seconds=2,
    description="basic straight line heuristic A* on Stanford map (4b-astar-2)",
)


grader.add_hidden_part(
    "4b-astar-3-hidden",
    lambda: t_4b_aStar(
        startLocation=locationFromTag(makeTag("landmark", "bookstore"), stanfordMap),
        heuristic=stanfordStraightLineHeuristic,
    ),
    max_points=0.5,
    max_seconds=2,
    description="hidden straight line heuristic A* on Stanford map (4b-astar-3)",
)


grader.add_hidden_part(
    "4b-astar-4-hidden",
    lambda: t_4b_aStar(
        startLocation=locationFromTag(makeTag("landmark", "evgr_a"), stanfordMap),
        heuristic=stanfordStraightLineHeuristic,
    ),
    max_points=1,
    max_seconds=2,
    description="hidden straight line heuristic A* on Stanford map (4b-astar-4)",
)


########################################################################################
# Problem 4c: "no waypoints" heuristic for A*


def t_4c_heuristic(
    startLocation: str, endTag: str, expectedCost: Optional[float] = None
):
    """Targeted test for `NoWaypointsHeuristic` -- uses the full Stanford map."""
    heuristic = submission.NoWaypointsHeuristic(endTag, stanfordMap)
    heuristicCost = heuristic.evaluate(util.State(startLocation))
    if expectedCost is not None:
        grader.require_is_equal(expectedCost, heuristicCost)



grader.add_basic_part(
    "4c-heuristic-1-basic",
    lambda: t_4c_heuristic(
        startLocation=locationFromTag(makeTag("landmark", "oval"), stanfordMap),
        endTag=makeTag("landmark", "gates"),
        expectedCost=446.99724421432353,
    ),
    max_points=1,
    max_seconds=2,
    description="basic no waypoints heuristic unit test",
)

grader.add_hidden_part(
    "4c-heuristic-1-hidden",
    lambda: t_4c_heuristic(
        startLocation=locationFromTag(makeTag("landmark", "bookstore"), stanfordMap),
        endTag=makeTag("amenity", "food"),
    ),
    max_points=1,
    max_seconds=2,
    description="hidden no waypoints heuristic unit test w/ multiple end locations",
)


# Initialize a `NoWaypointsHeuristic` using `endTag4c` and the `stanfordMap`
endTag4c = makeTag("wheelchair", "yes")
if grader.selectedPartName in [
    "4c-astar-1-basic",
    "4c-astar-2-basic",
    "4c-astar-3-hidden",
    "4c-astar-3-hidden",
    None,
]:
    try:
        stanfordNoWaypointsHeuristic = submission.NoWaypointsHeuristic(
            endTag4c, stanfordMap
        )
    except:
        stanfordNoWaypointsHeuristic = None

def t_4c_aStar(
    startLocation: str,
    waypointTags: List[str],
    heuristic: util.Heuristic,
    expectedCost: Optional[float] = None,
):
    """Run UCS on the A* Reduction of a WaypointsShortestPathProblem, w/ `heuristic`"""
    baseProblem = submission.WaypointsShortestPathProblem(
        startLocation, waypointTags, endTag4c, stanfordMap
    )
    aStarProblem = submission.aStarReduction(baseProblem, heuristic)

    # Solve the reduction via a call to `ucs.solve` (similar to prior tests)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(aStarProblem)
    path = extractPath(startLocation, ucs)
    grader.require_is_true(
        checkValid(path, stanfordMap, startLocation, endTag4c, waypointTags)
    )
    if expectedCost is not None:
        grader.require_is_equal(expectedCost, getTotalCost(path, stanfordMap))



grader.add_basic_part(
    "4c-astar-1-basic",
    lambda: t_4c_aStar(
        startLocation=locationFromTag(makeTag("landmark", "oval"), stanfordMap),
        waypointTags=[
            makeTag("landmark", "gates"),
            makeTag("landmark", "AOERC"),
            makeTag("landmark", "bookstore"),
            makeTag("landmark", "hoover_tower"),
        ],
        heuristic=stanfordNoWaypointsHeuristic,
        expectedCost=2943.242598551967,
    ),
    max_points=1.5,
    max_seconds=2,
    description="basic no waypoints heuristic A* on Stanford map (4c-astar-1)",
)


grader.add_basic_part(
    "4c-astar-2-basic",
    lambda: t_4c_aStar(
        startLocation=locationFromTag(makeTag("landmark", "AOERC"), stanfordMap),
        waypointTags=[
            makeTag("landmark", "tresidder"),
            makeTag("landmark", "hoover_tower"),
            makeTag("amenity", "food"),
        ],
        heuristic=stanfordNoWaypointsHeuristic,
        expectedCost=1677.3814380413373,
    ),
    max_points=1.5,
    max_seconds=2,
    description="basic no waypoints heuristic A* on Stanford map (4c-astar-2)",
)


grader.add_hidden_part(
    "4c-astar-3-hidden",
    lambda: t_4c_aStar(
        startLocation=locationFromTag(makeTag("landmark", "tresidder"), stanfordMap),
        waypointTags=[
            makeTag("landmark", "gates"),
            makeTag("amenity", "food"),
            makeTag("landmark", "rains"),
            makeTag("landmark", "stanford_stadium"),
            makeTag("bicycle", "yes"),
        ],
        heuristic=stanfordNoWaypointsHeuristic,
    ),
    max_points=3,
    max_seconds=10,
    description="hidden no waypoints heuristic A* on Stanford map (4c-astar-3)",
)

# Removed from Spring2025 per the Fixit doc's suggestion
# grader.add_manual_part("4d", max_points=2, description="example of n waypointTags")


if __name__ == "__main__":
    grader.grade()
