import doctest
import kornia_rs
import sys

def test_doctests():
    print("Running doctests on kornia_rs...")
    results = doctest.testmod(kornia_rs)
    print(f"Results: {results}")

    # Return exit code 1 if any test failed, so CI knows to fail
    if results.failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    test_doctests()
