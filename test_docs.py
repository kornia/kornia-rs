import doctest
import kornia_rs
import sys
import inspect

def test_doctests():
    print("Running doctests...")

    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(verbose=True)

    # Use a set to track object IDs we have already seen
    # This prevents testing 'resize' twice
    seen_ids = set()
    objects_to_test = []

    # Helper to add object safely
    def add_unique(obj):
        if id(obj) not in seen_ids:
            seen_ids.add(id(obj))
            objects_to_test.append(obj)

    # 1. Start with the main module
    add_unique(kornia_rs)

    # 2. Find all submodules
    submodules = []
    for name, obj in inspect.getmembers(kornia_rs):
        if inspect.ismodule(obj):
            print(f"Discovered submodule: {name}")
            submodules.append(obj)
            add_unique(obj)

    # 3. Look INSIDE submodules for Classes/Functions
    for module in submodules:
        print(f"Scanning submodule: {module.__name__}...")
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) or inspect.isroutine(obj):
                # Skip private members
                if name.startswith("_"):
                    continue
                # Add to test list (deduplication happens inside add_unique)
                add_unique(obj)

    total_failed = 0
    total_attempted = 0

    print(f"\nVerifying {len(objects_to_test)} unique objects...")

    for obj in objects_to_test:
        obj_name = getattr(obj, "__name__", str(obj))
        tests = finder.find(obj, name=obj_name)

        for test in tests:
            if not test.examples:
                continue

            print(f"Testing {test.name}...")
            runner.run(test)
            total_failed += runner.failures
            total_attempted += runner.tries

    print(f"Total: failed={total_failed}, attempted={total_attempted}")

    if total_failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    test_doctests()
