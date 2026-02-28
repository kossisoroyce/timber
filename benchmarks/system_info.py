from __future__ import annotations

import json
import platform
import sys


def main() -> None:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
