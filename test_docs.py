import doctest
import kornia_rs
import sys
import importlib

def test_doctests():
    print("Running doctests...")

    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(verbose=True)

    # 1. Start with the main module
    objects_to_test = [kornia_rs]

    # 2. Dynamically add submodules if they exist
    # We use importlib here to avoid UnboundLocalError
    try:
        # Attempt to load the apriltag submodule
        apriltag_mod = importlib.import_module("kornia_rs.apriltag")
        objects_to_test.append(apriltag_mod)
    except ImportError:
        # It's okay if it doesn't exist (e.g., feature disabled)
        pass

    total_failed = 0
    total_attempted = 0

    for obj in objects_to_test:
        # Find all tests in the object
        # getattr(obj, "__name__", "obj") gets the clean name of the module
        tests = finder.find(obj, name=getattr(obj, "__name__", "obj"))

        for test in tests:
            if not test.examples:
                continue

            print(f"Testing {test.name}...")
            runner.run(test)
            total_failed += runner.failures
            total_attempted += runner.tries

    print(f"Total: failed={total_failed}, attempted={total_attempted}")

    # Fail CI if any examples fail
    if total_failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    test_doctests()
