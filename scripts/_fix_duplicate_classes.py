"""Renames duplicate test class names — keeps first occurrence, renames later ones with B/C/D suffix."""

from pathlib import Path

target = Path("tests/test_session_2026_05_07_new_items.py")
lines = target.read_text(encoding="utf-8").splitlines(keepends=True)

# Find all class occurrences in order
seen = {}  # name -> count of occurrences so far
suffixes = ["B", "C", "D", "E"]

new_lines = []
for line in lines:
    if line.startswith("class Test") and line.rstrip().endswith(":"):
        name = line.strip().rstrip(":")
        if name not in seen:
            seen[name] = 0
            new_lines.append(line)
        else:
            suffix = suffixes[seen[name]]
            seen[name] += 1
            new_name = name + suffix
            new_line = line.replace(name, new_name, 1)
            new_lines.append(new_line)
        seen[name] = seen.get(name, 0)
        if name not in seen:
            seen[name] = 0
    else:
        new_lines.append(line)

# Recount properly
seen2 = {}
result = []
for line in lines:
    if line.startswith("class Test") and line.rstrip().endswith(":"):
        name = line.strip().rstrip(":")
        count = seen2.get(name, 0)
        seen2[name] = count + 1
        if count == 0:
            result.append(line)
        else:
            suffix = suffixes[count - 1]
            new_line = line.replace(name + ":", name + suffix + ":", 1)
            result.append(new_line)
            print(f"  Renamed: {name} -> {name + suffix}")
    else:
        result.append(line)

target.write_text("".join(result), encoding="utf-8")
print("Done. File written.")
