import doctest
import pytest
import kornia_rs

def run_doctests_recursive(module, runner, finder, visited):
    """Recursively finds and runs doctests in a module and its members."""
    if module in visited:
        return
    visited.add(module)

    # 1. Parse and Run doctests on the object itself
    # finder.find() returns a list of DocTest objects for the given object
    # We use name=... to give it a nice label in logs
    try:
        tests = finder.find(module, name=getattr(module, "__name__", str(module)))
        for test in tests:
            if test.examples:  # Only run if there are examples
                runner.run(test)
    except ValueError:
        # Some objects (like built-in types) might fail parsing, skip them
        pass

    # 2. Inspect all members (classes, functions, submodules)
    # dir() works on compiled modules where pkgutil.walk_packages fails
    for name in dir(module):
        if name.startswith("_"):
            continue

        try:
            obj = getattr(module, name)
        except Exception:
            continue

        # Check if object belongs to kornia_rs to avoid recursion into stdlib
        obj_mod = getattr(obj, "__module__", "")
        if hasattr(obj, "__name__") and (obj_mod and "kornia_rs" in obj_mod):
            # Recurse!
            run_doctests_recursive(obj, runner, finder, visited)

def test_compiled_docstrings():
    """
    Custom pytest hook to find and run doctests inside the compiled kornia_rs binary.
    """
    runner = doctest.DocTestRunner(verbose=True)
    finder = doctest.DocTestFinder()
    visited = set()

    print(f"\nScanning kornia_rs for doctests...")
    run_doctests_recursive(kornia_rs, runner, finder, visited)

    print(f"Ran {runner.tries} doctests.")

    # Fail if any doctests failed
    assert runner.failures == 0, f"Found {runner.failures} doctest failures!"
