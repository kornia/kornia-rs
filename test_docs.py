import doctest
import inspect
import sys
import kornia_rs

def get_testable_objects(mod):
    """
    Recursively find all classes and functions in the module
    that are part of kornia_rs.
    """
    objects = []
    seen = set()

    def recurse(obj, name_prefix):
        if id(obj) in seen:
            return
        seen.add(id(obj))

        # Add the object itself if it has a docstring
        if hasattr(obj, "__doc__") and obj.__doc__:
            objects.append((obj, name_prefix))

        # Inspect members
        for name, member in inspect.getmembers(obj):
            # Skip private/magic members
            if name.startswith("__"):
                continue

            # Filter to only keep objects belonging to kornia_rs
            if hasattr(member, "__module__") and member.__module__:
                 if "kornia_rs" not in member.__module__:
                     continue

            if inspect.ismodule(member) or inspect.isclass(member):
                recurse(member, f"{name_prefix}.{name}")

    recurse(mod, "kornia_rs")
    return objects

def main():
    print("Running doctests for kornia_rs...")

    # 1. Discovery
    items = get_testable_objects(kornia_rs)
    print(f"Discovered {len(items)} testable objects.")

    # 2. Test Execution
    # FIX: Use Parser + Runner manually to allow shared state
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(verbose=False)

    for obj, name in items:
        if not obj.__doc__:
            continue

        # Manually create the DocTest object
        # We pass 'name' as the filename too, just for error reporting
        test = parser.get_doctest(obj.__doc__, {"kornia_rs": kornia_rs}, name, name, 0)

        # Run it with our shared runner
        runner.run(test)

    # 3. Reporting
    failed = runner.failures
    attempted = runner.tries

    print("-" * 40)
    print(f"Test Summary: {attempted} examples attempted, {failed} failed.")
    print("-" * 40)

    if failed > 0:
        print("❌ Doctests failed!")
        sys.exit(1)
    else:
        print("✅ All doctests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
