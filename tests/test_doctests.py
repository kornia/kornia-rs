import doctest
import pytest
import kornia_rs
import inspect

def get_testable_objects(mod):
    """Recursively discover all objects with docstrings in the compiled module."""
    objects = []
    seen = set()

    def recurse(obj, name_prefix):
        if id(obj) in seen:
            return
        seen.add(id(obj))

        # 1. Add object if it has a docstring
        if hasattr(obj, "__doc__") and obj.__doc__:
            objects.append((obj, name_prefix))

        # 2. Inspect members
        for name in dir(obj):
            if name.startswith("_"):
                continue

            try:
                member = getattr(obj, name)
            except Exception:
                continue

            # Check if member belongs to kornia_rs
            mod_name = getattr(member, "__module__", None)
            if mod_name and "kornia_rs" in mod_name:
                if inspect.ismodule(member) or inspect.isclass(member):
                     recurse(member, f"{name_prefix}.{name}")

    recurse(mod, "kornia_rs")
    return objects

# Collect objects once
TEST_OBJECTS = get_testable_objects(kornia_rs)

@pytest.mark.parametrize("obj, name", TEST_OBJECTS)
def test_docstrings(obj, name):
    """Run doctest for a specific object."""
    # Create the context so 'kornia_rs' is available in the docstring examples
    globs = {"kornia_rs": kornia_rs}

    # Parser and Runner
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(verbose=False)

    # Create the test
    test = parser.get_doctest(obj.__doc__, globs, name, name, 0)

    # Run it
    runner.run(test)

    # Assert
    assert runner.failures == 0, f"Doctest failed for {name}. See output above."
