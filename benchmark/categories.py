"""Categories for spatial-QA questions.

Each category names a kind of spatial reasoning the question tests. Per-runner
accuracy is reported per category so we can identify exactly which question
types each runner (graph / vlm / hybrid) handles well.

The taxonomy is intentionally coarse. Adding a new category is cheap; splitting
or renaming is expensive because it invalidates per-category historical numbers.
"""

CATEGORIES: tuple[str, ...] = (
    "relative_position",   # left/right/above/below/in-front-of/behind queries
    "zone",                # location queries / wall-adjacency / region membership
    "proximity",           # near, next-to, adjacent
    "counting",            # how many X are there
    "multi_instance",      # the Nth X, all X, where are the X's
    "same_surface",        # items on the same physical surface as the anchor
    "containment",         # what is inside / on / under another object
    "out_of_view",         # answer is not visible in any single frame
    "negative",            # expected is empty (true-negative case)
    "ambiguity",           # multiple equally valid answers, no canonical pick
)
