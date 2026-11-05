#!/usr/bin/env bash
# =============================================================================
# Build the iSurgical GoogleTest suite with coverage, run every test executable, then write the
# pass-rate + coverage report to test/reports/.
#
#   test/run_tests.sh                  build, run, report
#   test/run_tests.sh --no-build       reuse build-test/ (just re-run and re-report)
#   BUILD_DIR=/path test/run_tests.sh  another build folder
#   JOBS=4 test/run_tests.sh           parallel compile jobs (default: all cores)
#
# ONNX Runtime / cuDNN folders: taken from $ONNXRUNTIME_ROOT_DIR / $CUDNN_LIB_DIR if set,
# otherwise from the main build's build/CMakeCache.txt (the values you configured the project with).
#
# Output:
#   test/reports/test_report.html     pass rate + coverage, with annotated source
#   test/reports/test_report.md       same summary as Markdown
#   test/reports/summary.json         numbers for CI
#   test/reports/xml/<suite>.xml      raw GoogleTest results (JUnit format)
#   test/reports/logs/<suite>.log     full output of each test executable
#
# Exit code: 0 if every test passed, 1 otherwise (the report is written either way).
# =============================================================================

# Stop on errors, unset variables and failures inside pipes
set -euo pipefail

# <project>/test
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root
ROOT="$(dirname "$TEST_DIR")"
# Build folder
BUILD="${BUILD_DIR:-$ROOT/build-test}"
# Reports
REPORT="$TEST_DIR/reports"
# Compile jobs
JOBS="${JOBS:-$(nproc)}"
# Build unless --no-build
DO_BUILD=1
# Only option
if [[ "${1:-}" == "--no-build" ]]; then DO_BUILD=0; fi

# Read one cache entry (e.g. ONNXRUNTIME_ROOT_DIR) from the main build, "" if missing
cache_value() {
    # build/CMakeCache.txt lines look like NAME:PATH=value
    grep -E "^$1:[A-Z]+=" "$ROOT/build/CMakeCache.txt" 2>/dev/null | head -1 | cut -d= -f2- || true
}
# ONNX Runtime folder: environment first, then the main build
ORT_DIR="${ONNXRUNTIME_ROOT_DIR:-$(cache_value ONNXRUNTIME_ROOT_DIR)}"
# cuDNN folder: environment first, then the main build (may stay empty: tests run on CPU)
CUDNN_DIR="${CUDNN_LIB_DIR:-$(cache_value CUDNN_LIB_DIR)}"

# Build step
if [[ $DO_BUILD -eq 1 ]]; then
    # ONNX Runtime is required
    if [[ -z "$ORT_DIR" ]]; then
        # Explain how to fix it
        echo "ONNXRUNTIME_ROOT_DIR is not set and build/CMakeCache.txt has none; e.g." >&2
        # Example
        echo "  ONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime-linux-x64-gpu-1.20.1 test/run_tests.sh" >&2
        # Stop
        exit 2
    fi
    # Configure
    echo "==> configure ($BUILD)"
    # Stand-alone test project; the main CMakeLists.txt is untouched
    cmake -S "$TEST_DIR" -B "$BUILD" -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON \
          -DONNXRUNTIME_ROOT_DIR="$ORT_DIR" -DCUDNN_LIB_DIR="$CUDNN_DIR" > /dev/null
    # Compile
    echo "==> build (-j$JOBS)"
    # Only warnings / errors are shown
    cmake --build "$BUILD" -j"$JOBS" 2>&1 | grep -Ev '^\[ *[0-9]+%\]|^-- |Built target|Scanning|Generating fake' || true
    # The grep above hides the exit code: build again (no-op if it succeeded) to fail on errors
    cmake --build "$BUILD" -j"$JOBS" > /dev/null
fi

# Old coverage counters out, so the report reflects this run only
find "$BUILD" -name '*.gcda' -delete
# Fresh reports
rm -rf "$REPORT"
# XML + logs folders
mkdir -p "$REPORT/xml" "$REPORT/logs"
# Fresh scratch folder
rm -rf "$BUILD/tmp"
# gtest's temporary files
mkdir -p "$BUILD/tmp"
# Tell gtest
export TEST_TMPDIR="$BUILD/tmp"

# 0 = all passed
STATUS=0
# Run every test executable
echo "==> run"
# Sorted for a stable order
for exe in $(find "$BUILD" -maxdepth 1 -type f -name 'test_*' -perm -u+x | sort); do
    # Suite name
    name="$(basename "$exe")"
    # Start time
    start=$SECONDS
    # XML for the report, everything else to the log
    if "$exe" --gtest_output="xml:$REPORT/xml/$name.xml" > "$REPORT/logs/$name.log" 2>&1; then
        # Passed
        result="PASS"
    else
        # Failed: keep going with the other suites
        result="FAIL"
        # Overall failure
        STATUS=1
    fi
    # Skipped tests in this suite (real models missing), shown next to the result
    skipped=$(grep -c '^\[  SKIPPED \] [A-Za-z]' "$REPORT/logs/$name.log" || true)
    # One line per suite
    printf "  %-4s  %-24s (%ds)%s\n" "$result" "$name" "$((SECONDS - start))" \
           "$([[ $skipped -gt 0 ]] && echo "  $skipped skipped")"
done

# Reports
echo "==> report"
# Isolated mode: nothing from the current folder is imported
python3 -I "$TEST_DIR/tools/gen_report.py" --root "$ROOT" --build "$BUILD" --reports "$REPORT"

# 0 = every test passed
exit $STATUS
