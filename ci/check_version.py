"""Guard a release tag against the packaged version.

Used by the release workflow: given the pushed tag (e.g. ``v0.7.0``), confirm
its version matches ``setup.py`` and refuse to publish on a mismatch (which
would otherwise upload the wrong version under the tag). Also reports whether
the version is a PEP 440 pre-release so the GitHub release can be flagged.
"""

import os
import re
import sys

from packaging.version import Version


def packaged_version():
    with open("setup.py", encoding="utf-8") as fh:
        source = fh.read()
    match = re.search(r'\bversion\s*=\s*"([^"]+)"', source)
    if match is None:
        sys.exit("::error::could not find version= in setup.py")
    return match.group(1)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: check_version.py <tag>")
    tag = sys.argv[1]
    tag_version = tag[1:] if tag.startswith("v") else tag
    pkg_version = packaged_version()

    if tag_version != pkg_version:
        sys.exit(
            f"::error::tag {tag} (version {tag_version}) does not match "
            f"setup.py version {pkg_version}"
        )

    prerelease = Version(tag_version).is_prerelease
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"prerelease={'true' if prerelease else 'false'}\n")
    print(f"OK: releasing {pkg_version} (prerelease={prerelease})")


if __name__ == "__main__":
    main()
