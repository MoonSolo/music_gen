def parse_prompt_file(filepath):
    """
    Parses a prompt file and returns a list of dicts.
    Each dict has: prompt, duration, filename
    """
    entries = []
    errors = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            parts = [p.strip() for p in line.split('|')]

            if len(parts) != 3:
                errors.append(f"Line {line_num}: expected 3 fields separated by '|', got {len(parts)}: '{line}'")
                continue

            prompt, duration_str, filename = parts

            if not prompt:
                errors.append(f"Line {line_num}: prompt is empty")
                continue

            try:
                duration = int(duration_str)
                if duration < 5 or duration > 120:
                    raise ValueError
            except ValueError:
                errors.append(f"Line {line_num}: duration must be an integer between 5 and 120, got '{duration_str}'")
                continue

            if not filename:
                errors.append(f"Line {line_num}: filename is empty")
                continue

            entries.append({
                'prompt': prompt,
                'duration': duration,
                'filename': filename,
                'line_num': line_num
            })

    return entries, errors


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "prompts.txt"

    entries, errors = parse_prompt_file(filepath)

    if errors:
        print(f"\n⚠ Found {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")

    print(f"\n✓ {len(entries)} valid prompt(s) found:")
    for e in entries:
        print(f"  [{e['line_num']}] {e['filename']} | {e['duration']}s | {e['prompt']}")