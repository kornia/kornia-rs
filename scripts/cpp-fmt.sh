#!/bin/bash
# Helper script for C++ formatting
# Usage: cpp-fmt.sh [--check]
#   --check    Only check formatting, don't modify files

set -e

CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --check) CHECK_ONLY=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Find C++ files, excluding build directories
FILES=$(find include src tests examples -type f \( -name '*.cpp' -o -name '*.hpp' \) -not -path '*/build/*' 2>/dev/null)

if [ -z "$FILES" ]; then
    echo "No C++ files found"
    exit 0
fi

if [ "$CHECK_ONLY" = true ]; then
    echo "Checking C++ formatting..."
    echo "$FILES" | xargs clang-format --dry-run --Werror
    echo "All files formatted correctly!"
else
    echo "Formatting C++ files..."
    echo "$FILES" | xargs clang-format -i
    echo "Done!"
fi
