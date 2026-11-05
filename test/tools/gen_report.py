#!/usr/bin/env python3
"""Turn GoogleTest XML results + gcov data into a pass-rate and coverage report.

    python3 test/tools/gen_report.py --root <project> --build <build-test> --reports <test/reports>

Writes summary.json, test_report.md and test_report.html into --reports.
Only the standard library and the gcov that ships with gcc are used (no lcov / gcovr needed).
"""

# Command-line options
import argparse
# Parallel gcov calls
import concurrent.futures
# HTML escaping of source code and messages
import html
# gcov --json-format output, summary.json
import json
# Paths
import os
# Running gcov
import subprocess
# Exit with an error code
import sys
# Report timestamp
from datetime import datetime
# Reading the gtest XML files
import xml.etree.ElementTree as ET

# Project source files measured for coverage (relative to the project root): prefix match
COVERED_PREFIXES = ("src/",)
# Project files that the test build does not compile, and why (listed in the report for transparency)
EXCLUDED = {
    # Command-line entry point: 10 lines that call iSurgicalConfig::FromArgs and iSurgical::Run, both tested
    "src/main.cpp": "entry point (wiring only); FromArgs / PrintUsage / iSurgical::Run are tested directly",
}


# ---------------------------------------------------------------------------
# Pass rate: GoogleTest XML
# ---------------------------------------------------------------------------

def parse_gtest_xml(xml_dir):
    """Return a list of suites: {executable, name, tests: [{name, status, time, message}]}."""
    # One entry per <testsuite>
    suites = []
    # Every XML file written by --gtest_output
    for file_name in sorted(os.listdir(xml_dir)) if os.path.isdir(xml_dir) else []:
        # Only .xml
        if not file_name.endswith(".xml"):
            # Skip anything else
            continue
        # Executable name, e.g. test_console
        executable = file_name[:-4]
        # Parse; a truncated file (crashed run) is reported as one failure
        try:
            # Root element <testsuites>
            root = ET.parse(os.path.join(xml_dir, file_name)).getroot()
        # Broken XML
        except ET.ParseError as error:
            # Synthetic failed suite so the crash is visible
            suites.append({"executable": executable, "name": executable,
                           "tests": [{"name": "(xml)", "status": "failed", "time": 0.0, "message": str(error)}]})
            # Next file
            continue
        # Each <testsuite>
        for suite in root.iter("testsuite"):
            # Test cases of this suite
            tests = []
            # Each <testcase>
            for case in suite.iter("testcase"):
                # <failure> children (one per failed assertion)
                failures = case.findall("failure")
                # <skipped> child
                skipped = case.find("skipped") is not None
                # Classify
                if failures:
                    # At least one assertion failed
                    status = "failed"
                # Explicit GTEST_SKIP or not run
                elif skipped or case.get("status") == "notrun" or case.get("result") == "skipped":
                    # Not counted as passed or failed
                    status = "skipped"
                # Otherwise
                else:
                    # Passed
                    status = "passed"
                # First failure message (trimmed) for the report
                message = (failures[0].get("message") or "").strip() if failures else ""
                # Store
                tests.append({"name": case.get("name"), "status": status,
                              "time": float(case.get("time", "0") or 0), "message": message})
            # Store the suite
            suites.append({"executable": executable, "name": suite.get("name"), "tests": tests})
    # All suites
    return suites


def pass_rate_summary(suites):
    """Totals per executable and overall."""
    # Per executable: {name: {total, passed, failed, skipped, time}}
    per_exe = {}
    # Each suite
    for suite in suites:
        # Entry for its executable
        entry = per_exe.setdefault(suite["executable"],
                                   {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "time": 0.0, "suites": 0})
        # One more suite
        entry["suites"] += 1
        # Each test
        for test in suite["tests"]:
            # Count it
            entry["total"] += 1
            # By status
            entry[test["status"]] += 1
            # Time
            entry["time"] += test["time"]
    # Overall totals
    total = {key: sum(e[key] for e in per_exe.values()) for key in ("total", "passed", "failed", "skipped", "time")}
    # Tests that actually ran (skipped ones do not count against the rate)
    executed = total["passed"] + total["failed"]
    # Pass rate in percent
    total["pass_rate"] = 100.0 * total["passed"] / executed if executed else 0.0
    # Per-executable pass rate
    for entry in per_exe.values():
        # Executed in this executable
        ran = entry["passed"] + entry["failed"]
        # Percent
        entry["pass_rate"] = 100.0 * entry["passed"] / ran if ran else 0.0
    # Both
    return per_exe, total


# ---------------------------------------------------------------------------
# Coverage: gcov --json-format
# ---------------------------------------------------------------------------

def run_gcov(gcda):
    """Run gcov on one .gcda file and return its parsed JSON (or None)."""
    # --json-format + --stdout: one JSON document on stdout; --branch-probabilities: include per-line branch data
    # (without it the JSON has no "branches"); --object-directory: where the .gcno lives
    result = subprocess.run(["gcov", "--json-format", "--stdout", "--branch-probabilities",
                             "--object-directory", os.path.dirname(gcda), gcda],
                            cwd=os.path.dirname(gcda), capture_output=True, text=True, check=False)
    # gcov failed (e.g. stale .gcno)
    if result.returncode != 0 or not result.stdout.strip():
        # Ignore this object
        return None
    # Parse
    return json.loads(result.stdout)


def collect_coverage(root, build):
    """Merge every .gcda in the build folder into per-file line / function / branch data."""
    # All counter files (the tests' scratch folder has none)
    gcda_files = []
    # Walk the build tree
    for folder, _, files in os.walk(build):
        # Each counter file
        gcda_files.extend(os.path.join(folder, f) for f in files if f.endswith(".gcda"))
    # file → {"lines": {n: count}, "functions": {start_line: {...}}, "branches": {(line, fn, i): taken}}
    files = {}
    # Run gcov in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        # Each JSON result
        for data in pool.map(run_gcov, gcda_files):
            # Skip failures
            if not data:
                # Next
                continue
            # Each source file mentioned by this object
            for entry in data.get("files", []):
                # Absolute, normalised path
                path = os.path.realpath(os.path.join(data.get("current_working_directory", ""), entry["file"]))
                # Relative to the project root
                rel = os.path.relpath(path, root)
                # Only project files we measure
                if not rel.startswith(COVERED_PREFIXES):
                    # System headers, Boost, gtest, the tests themselves
                    continue
                # Merged record for this file
                record = files.setdefault(rel, {"lines": {}, "functions": {}, "branches": {}})
                # Each executable line (templates / inline functions can appear several times)
                for line in entry.get("lines", []):
                    # Line number
                    number = line["line_number"]
                    # Sum the counts of every instance
                    record["lines"][number] = record["lines"].get(number, 0) + line["count"]
                    # Each branch on this line
                    for index, branch in enumerate(line.get("branches", [])):
                        # Exception edges are not counted (same as gcovr --exclude-throw-branches)
                        if branch.get("throw"):
                            # Skip
                            continue
                        # Key: line, function instance, branch index
                        key = (number, line.get("function_name", ""), index)
                        # Taken in any run
                        record["branches"][key] = record["branches"].get(key, False) or branch["count"] > 0
                # Each function (merged by start line, so template instances count once)
                for fn in entry.get("functions", []):
                    # Existing record for this start line
                    current = record["functions"].get(fn["start_line"])
                    # Keep the readable name and the "called at least once" flag
                    record["functions"][fn["start_line"]] = {
                        "name": fn.get("demangled_name") or fn["name"],
                        "called": (current["called"] if current else False) or fn["execution_count"] > 0,
                    }
    # Per-file data
    return files


def coverage_summary(files):
    """Line / function / branch totals per file and overall."""
    # Per-file rows
    rows = []
    # Each file, sorted by path
    for rel in sorted(files):
        # Its record
        record = files[rel]
        # Executable lines
        lines_total = len(record["lines"])
        # Skip files without executable code (e.g. headers with only declarations)
        if lines_total == 0:
            # Next
            continue
        # Lines run at least once
        lines_hit = sum(1 for c in record["lines"].values() if c > 0)
        # Functions
        fn_total = len(record["functions"])
        # Functions called
        fn_hit = sum(1 for f in record["functions"].values() if f["called"])
        # Branches
        br_total = len(record["branches"])
        # Branches taken
        br_hit = sum(1 for taken in record["branches"].values() if taken)
        # Line numbers never run
        missed = sorted(n for n, c in record["lines"].items() if c == 0)
        # Row
        rows.append({"file": rel, "lines": [lines_hit, lines_total], "functions": [fn_hit, fn_total],
                     "branches": [br_hit, br_total], "missed_lines": missed,
                     "missed_functions": sorted(f["name"] for f in record["functions"].values() if not f["called"])})
    # Overall sums
    total = {kind: [sum(r[kind][0] for r in rows), sum(r[kind][1] for r in rows)]
             for kind in ("lines", "functions", "branches")}
    # Rows and totals
    return rows, total


def pct(pair):
    """[hit, total] → percentage (100 when there is nothing to cover)."""
    # Avoid division by zero
    return 100.0 * pair[0] / pair[1] if pair[1] else 100.0


def ranges(numbers):
    """[1,2,3,7,9,10] → "1-3, 7, 9-10" for compact missed-line lists."""
    # Output pieces
    parts = []
    # Current run start / end
    start = prev = None
    # Each number (sorted)
    for n in numbers:
        # Continues the run
        if prev is not None and n == prev + 1:
            # Extend
            prev = n
            # Next
            continue
        # Close the previous run
        if start is not None:
            # "a" or "a-b"
            parts.append(str(start) if start == prev else f"{start}-{prev}")
        # Start a new run
        start = prev = n
    # Close the last run
    if start is not None:
        # "a" or "a-b"
        parts.append(str(start) if start == prev else f"{start}-{prev}")
    # Joined
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_markdown(path, generated, per_exe, total, suites, rows, cov_total):
    """test_report.md"""
    # Lines of the document
    out = [f"# iSurgical — Test Report", "", f"Generated {generated}", ""]
    # Headline numbers
    out += ["## Summary", "", "| Metric | Result |", "|---|---|",
            f"| Pass rate | **{total['pass_rate']:.1f}%** ({total['passed']}/{total['passed'] + total['failed']}) |",
            f"| Tests | {total['total']} total · {total['passed']} passed · {total['failed']} failed · "
            f"{total['skipped']} skipped |",
            f"| Line coverage | **{pct(cov_total['lines']):.1f}%** ({cov_total['lines'][0]}/{cov_total['lines'][1]}) |",
            f"| Function coverage | {pct(cov_total['functions']):.1f}% "
            f"({cov_total['functions'][0]}/{cov_total['functions'][1]}) |",
            f"| Branch coverage | {pct(cov_total['branches']):.1f}% "
            f"({cov_total['branches'][0]}/{cov_total['branches'][1]}) |",
            f"| Test time | {total['time']:.1f} s |", ""]
    # Per executable
    out += ["## Pass rate by test executable", "", "| Executable | Tests | Passed | Failed | Skipped | Pass rate | Time |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    # Each executable
    for name, e in sorted(per_exe.items()):
        # One row
        out.append(f"| `{name}` | {e['total']} | {e['passed']} | {e['failed']} | {e['skipped']} | "
                   f"{e['pass_rate']:.1f}% | {e['time']:.1f} s |")
    # Failures, if any
    failed = [(s, t) for s in suites for t in s["tests"] if t["status"] == "failed"]
    # Section
    out += ["", "## Failed tests", ""]
    # List or "none"
    out += [f"- `{s['name']}.{t['name']}` — {t['message'].splitlines()[0] if t['message'] else ''}"
            for s, t in failed] or ["None."]
    # Coverage per file
    out += ["", "## Coverage by source file", "", "| File | Lines | Functions | Branches | Missed lines |",
            "|---|---:|---:|---:|---|"]
    # Each file
    for r in rows:
        # One row
        out.append(f"| `{r['file']}` | {pct(r['lines']):.1f}% ({r['lines'][0]}/{r['lines'][1]}) | "
                   f"{pct(r['functions']):.1f}% | {pct(r['branches']):.1f}% | {ranges(r['missed_lines']) or '—'} |")
    # Excluded files
    out += ["", "## Not measured", ""] + [f"- `{f}` — {why}" for f, why in EXCLUDED.items()]
    # Notes
    out += ["", "Branch coverage excludes exception-only edges. Line numbers refer to the source files."]
    # Write
    with open(path, "w", encoding="utf-8") as fh:
        # Joined with newlines
        fh.write("\n".join(out) + "\n")


def bar(percent):
    """Small inline progress bar for the HTML tables."""
    # Colour class by threshold
    level = "good" if percent >= 90 else "ok" if percent >= 75 else "low"
    # Track + fill
    return (f'<span class="bar"><span class="fill {level}" style="width:{percent:.1f}%"></span></span>'
            f'<span class="num">{percent:.1f}%</span>')


def write_html(path, generated, root, per_exe, total, suites, rows, cov_total, files):
    """test_report.html: summary, per-executable and per-file tables, failures, annotated source."""
    # Shorthand
    esc = html.escape
    # Headline tiles
    tiles = [("Pass rate", f"{total['pass_rate']:.1f}%", f"{total['passed']} of {total['passed'] + total['failed']} tests"),
             ("Line coverage", f"{pct(cov_total['lines']):.1f}%", f"{cov_total['lines'][0]} of {cov_total['lines'][1]} lines"),
             ("Function coverage", f"{pct(cov_total['functions']):.1f}%",
              f"{cov_total['functions'][0]} of {cov_total['functions'][1]} functions"),
             ("Branch coverage", f"{pct(cov_total['branches']):.1f}%",
              f"{cov_total['branches'][0]} of {cov_total['branches'][1]} branches")]
    # Tile HTML
    tiles_html = "".join(f'<div class="tile"><div class="label">{esc(a)}</div><div class="value">{esc(b)}</div>'
                         f'<div class="sub">{esc(c)}</div></div>' for a, b, c in tiles)
    # Per-executable rows
    exe_rows = "".join(
        f"<tr><td><code>{esc(n)}</code></td><td>{e['total']}</td><td>{e['passed']}</td>"
        f"<td class=\"{'bad' if e['failed'] else ''}\">{e['failed']}</td><td>{e['skipped']}</td>"
        f"<td>{bar(e['pass_rate'])}</td><td>{e['time']:.1f} s</td></tr>"
        for n, e in sorted(per_exe.items()))
    # Every test case, grouped by suite, as collapsible lists
    suite_blocks = []
    # Each suite
    for s in suites:
        # Counts for the summary line
        passed = sum(1 for t in s["tests"] if t["status"] == "passed")
        # Items
        items = "".join(
            f'<li class="{t["status"]}"><span class="dot"></span>{esc(t["name"])}'
            f'<span class="time">{t["time"] * 1000:.0f} ms</span>'
            + (f'<pre>{esc(t["message"])}</pre>' if t["message"] else "") + "</li>"
            for t in s["tests"])
        # Open automatically if something failed
        opened = " open" if passed < len(s["tests"]) else ""
        # Block
        suite_blocks.append(f'<details{opened}><summary><code>{esc(s["name"])}</code> '
                            f'<span class="muted">({esc(s["executable"])}) · {passed}/{len(s["tests"])} passed</span>'
                            f'</summary><ul class="cases">{items}</ul></details>')
    # Per-file coverage rows
    file_rows = "".join(
        f'<tr><td><a href="#src-{i}"><code>{esc(r["file"])}</code></a></td><td>{bar(pct(r["lines"]))}'
        f'<span class="frac">{r["lines"][0]}/{r["lines"][1]}</span></td><td>{bar(pct(r["functions"]))}</td>'
        f'<td>{bar(pct(r["branches"]))}</td></tr>' for i, r in enumerate(rows))
    # Annotated source per file
    sources = []
    # Each measured file
    for i, r in enumerate(rows):
        # Line counts for this file
        counts = files[r["file"]]["lines"]
        # Source text (may be missing if the file moved)
        try:
            # Read it
            with open(os.path.join(root, r["file"]), encoding="utf-8", errors="replace") as fh:
                # All lines
                text_lines = fh.read().splitlines()
        # Missing
        except OSError:
            # Nothing to show
            text_lines = []
        # Rendered lines
        body = []
        # Each line
        for n, text in enumerate(text_lines, start=1):
            # Executable and run / executable and missed / not executable
            cls = ("hit" if counts[n] > 0 else "miss") if n in counts else ""
            # Count column
            count = str(counts[n]) if n in counts else ""
            # Row
            body.append(f'<tr class="{cls}"><td class="ln">{n}</td><td class="cnt">{count}</td>'
                        f'<td class="code">{esc(text) or " "}</td></tr>')
        # Missed functions note
        missed_fn = (f'<p class="muted">Never called: {esc(", ".join(r["missed_functions"]))}</p>'
                     if r["missed_functions"] else "")
        # Collapsible block
        sources.append(f'<details id="src-{i}"><summary><code>{esc(r["file"])}</code> '
                       f'<span class="muted">{pct(r["lines"]):.1f}% lines · missed: '
                       f'{esc(ranges(r["missed_lines"]) or "none")}</span></summary>{missed_fn}'
                       f'<div class="src"><table>{"".join(body)}</table></div></details>')
    # Excluded list
    excluded = "".join(f"<li><code>{esc(f)}</code> — {esc(why)}</li>" for f, why in EXCLUDED.items())
    # Whole page
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Test Report</title>
<style>
:root {{ --bg:#fbfbfa; --panel:#ffffff; --text:#1f2328; --muted:#6a6f76; --line:#e4e4e1;
  --good:#2f8f5b; --ok:#b7861b; --low:#c0442f; --hit:#e8f5ec; --miss:#fbe9e6; --code:#f4f4f2; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#16181b; --panel:#1e2125; --text:#e6e6e3;
  --muted:#9a9fa6; --line:#30343a; --good:#4cbb7f; --ok:#d9a93a; --low:#e5664f; --hit:#1d3326; --miss:#3a2220;
  --code:#23272c; }} }}
:root[data-theme="dark"] {{ --bg:#16181b; --panel:#1e2125; --text:#e6e6e3; --muted:#9a9fa6; --line:#30343a;
  --good:#4cbb7f; --ok:#d9a93a; --low:#e5664f; --hit:#1d3326; --miss:#3a2220; --code:#23272c; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:32px 16px 64px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:36px 0 12px; }}
.muted {{ color:var(--muted); font-size:13px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-top:20px; }}
.tile {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.tile .label {{ color:var(--muted); font-size:13px; }} .tile .value {{ font-size:28px; font-weight:650; }}
.tile .sub {{ color:var(--muted); font-size:12px; }}
.scroll {{ overflow-x:auto; }}
table.grid {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:10px; }}
table.grid th, table.grid td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
table.grid th {{ font-size:12px; color:var(--muted); font-weight:600; }}
td.bad {{ color:var(--low); font-weight:650; }}
.bar {{ display:inline-block; width:80px; height:6px; border-radius:3px; background:var(--line); vertical-align:middle; margin-right:8px; overflow:hidden; }}
.fill {{ display:block; height:100%; }} .fill.good {{ background:var(--good); }} .fill.ok {{ background:var(--ok); }} .fill.low {{ background:var(--low); }}
.num {{ font-variant-numeric:tabular-nums; }} .frac {{ color:var(--muted); font-size:12px; margin-left:6px; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; }}
a {{ color:inherit; }}
details {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; margin:8px 0; }}
summary {{ cursor:pointer; padding:10px 12px; }}
ul.cases {{ list-style:none; margin:0; padding:0 12px 10px; }}
ul.cases li {{ padding:3px 0; font-size:13px; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; background:var(--good); }}
li.failed .dot {{ background:var(--low); }} li.skipped .dot {{ background:var(--muted); }}
.time {{ color:var(--muted); float:right; font-variant-numeric:tabular-nums; }}
pre {{ background:var(--code); padding:8px; border-radius:6px; white-space:pre-wrap; font-size:12px; }}
.src {{ overflow-x:auto; border-top:1px solid var(--line); }}
.src table {{ border-collapse:collapse; width:100%; font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }}
.src td {{ padding:0 8px; white-space:pre; }} .src .ln, .src .cnt {{ color:var(--muted); text-align:right; user-select:none; }}
.src tr.hit {{ background:var(--hit); }} .src tr.miss {{ background:var(--miss); }}
</style></head><body><main>
<h1>iSurgical — Test Report</h1>
<div class="muted">Generated {esc(generated)} · GoogleTest {total['total']} tests in {len(per_exe)} executables · {total['time']:.1f} s</div>
<div class="tiles">{tiles_html}</div>
<h2>Pass rate by test executable</h2>
<div class="scroll"><table class="grid"><tr><th>Executable</th><th>Tests</th><th>Passed</th><th>Failed</th><th>Skipped</th><th>Pass rate</th><th>Time</th></tr>{exe_rows}</table></div>
<h2>Coverage by source file</h2>
<div class="scroll"><table class="grid"><tr><th>File</th><th>Lines</th><th>Functions</th><th>Branches</th></tr>{file_rows}</table></div>
<p class="muted">Branch coverage excludes exception-only edges. Not measured:</p><ul class="muted">{excluded}</ul>
<h2>Test cases</h2>{"".join(suite_blocks)}
<h2>Annotated source</h2><p class="muted">Green = executed, red = never executed; the second column is the execution count.</p>
{"".join(sources)}
</main></body></html>
"""
    # Write
    with open(path, "w", encoding="utf-8") as fh:
        # Page
        fh.write(page)


def main():
    """Entry point."""
    # Options
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Project root
    parser.add_argument("--root", required=True)
    # Build folder with .gcda/.gcno files
    parser.add_argument("--build", required=True)
    # Report folder (contains xml/)
    parser.add_argument("--reports", required=True)
    # Parse
    args = parser.parse_args()
    # Normalise paths
    root, build, reports = (os.path.realpath(p) for p in (args.root, args.build, args.reports))
    # Timestamp
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Pass rate
    suites = parse_gtest_xml(os.path.join(reports, "xml"))
    # Totals
    per_exe, total = pass_rate_summary(suites)
    # Coverage
    files = collect_coverage(root, build)
    # Totals
    rows, cov_total = coverage_summary(files)

    # Machine-readable summary
    summary = {"generated": generated, "tests": total, "per_executable": per_exe,
               "coverage": {k: {"hit": v[0], "total": v[1], "percent": round(pct(v), 2)} for k, v in cov_total.items()},
               "files": rows, "excluded": EXCLUDED}
    # Write it
    with open(os.path.join(reports, "summary.json"), "w", encoding="utf-8") as fh:
        # Pretty
        json.dump(summary, fh, indent=2)
    # Markdown
    write_markdown(os.path.join(reports, "test_report.md"), generated, per_exe, total, suites, rows, cov_total)
    # HTML
    write_html(os.path.join(reports, "test_report.html"), generated, root, per_exe, total, suites, rows, cov_total,
               files)

    # Console summary
    print(f"  tests     : {total['total']} total, {total['passed']} passed, {total['failed']} failed, "
          f"{total['skipped']} skipped  ->  pass rate {total['pass_rate']:.1f}%")
    # Coverage line
    print(f"  coverage  : lines {pct(cov_total['lines']):.1f}%  functions {pct(cov_total['functions']):.1f}%  "
          f"branches {pct(cov_total['branches']):.1f}%")
    # Where to look
    print(f"  report    : {os.path.join(reports, 'test_report.html')}")
    # Non-zero if nothing was found (misconfigured run)
    return 0 if suites else 1


# Run when executed as a script
if __name__ == "__main__":
    # Exit with main's code
    sys.exit(main())
