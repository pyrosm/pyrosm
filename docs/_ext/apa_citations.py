"""APA-style citations matching the Python GIS book documentation."""

from collections import Counter
from dataclasses import dataclass, field

import pybtex.plugin
import sphinxcontrib.bibtex.plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.labels import BaseLabelStyle
from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year import (
    AuthorYearReferenceStyle,
)


class APALabelStyle(BaseLabelStyle):
    """Format labels as ``Author, year`` with APA-style disambiguation."""

    def format_labels(self, sorted_entries):
        """Yield one label per entry, adding a, b, ... to labels that repeat."""
        labels = [self.format_label(entry) for entry in sorted_entries]
        label_counts = Counter(labels)
        used_counts = Counter()

        for label in labels:
            if label_counts[label] == 1:
                yield label
            else:
                yield label + chr(ord("a") + used_counts[label])
                used_counts.update([label])

    def format_label(self, entry):
        """Return ``Author, year`` with the names the entry's type uses."""
        if entry.type in {"book", "inbook"}:
            label = self.author_editor_key_label(entry)
        elif entry.type == "proceedings":
            label = self.editor_key_organization_label(entry)
        elif entry.type == "manual":
            label = self.author_key_organization_label(entry)
        else:
            label = self.author_key_label(entry)

        if "year" in entry.fields:
            return f"{label}, {entry.fields['year']}"
        return label

    def author_key_label(self, entry):
        """Return the authors' names, else the first three characters of the key."""
        if "author" in entry.persons:
            return self.format_names(entry.persons["author"])
        if "key" in entry.fields:
            return entry.fields["key"][:3]
        return entry.key[:3]

    def author_editor_key_label(self, entry):
        """Return the authors' names, else the editors', else the key's start."""
        if "author" in entry.persons:
            return self.format_names(entry.persons["author"])
        if "editor" in entry.persons:
            return self.format_names(entry.persons["editor"])
        if "key" in entry.fields:
            return entry.fields["key"][:3]
        return entry.key[:3]

    def author_key_organization_label(self, entry):
        """Return the authors' names, else the key's start, else the organization."""
        if "author" in entry.persons:
            return self.format_names(entry.persons["author"])
        if "key" in entry.fields:
            return entry.fields["key"][:3]
        if "organization" in entry.fields:
            return self.format_organization(entry.fields["organization"])
        return entry.key[:3]

    def editor_key_organization_label(self, entry):
        """Return the editors' names, else the key's start, else the organization."""
        if "editor" in entry.persons:
            return self.format_names(entry.persons["editor"])
        if "key" in entry.fields:
            return entry.fields["key"][:3]
        if "organization" in entry.fields:
            return self.format_organization(entry.fields["organization"])
        return entry.key[:3]

    @staticmethod
    def format_organization(organization):
        """Return the organization's name without a leading ``The``."""
        if organization.startswith("The "):
            return organization[4:]
        return organization

    @staticmethod
    def format_names(persons):
        """Return one surname, two joined by ``&``, or the first and ``et al.``."""
        family_names = [
            " ".join(person.prelast_names + person.last_names) for person in persons
        ]

        if not family_names:
            return ""
        if len(family_names) == 1:
            return family_names[0]
        if len(family_names) == 2:
            return f"{family_names[0]} & {family_names[1]}"
        return f"{family_names[0]} et al."


class APAStyle(UnsrtStyle):
    """Use author-year labels and alphabetical order in the bibliography."""

    default_label_style = APALabelStyle
    default_sorting_style = "author_year_title"


def no_brackets() -> BracketStyle:
    """Leave punctuation to the surrounding prose."""

    return BracketStyle(left="", right="")


@dataclass
class AuthorYearNoBrackets(AuthorYearReferenceStyle):
    """Render author-year citations without automatically adding brackets."""

    bracket_parenthetical: BracketStyle = field(default_factory=no_brackets)
    bracket_textual: BracketStyle = field(default_factory=no_brackets)
    bracket_author: BracketStyle = field(default_factory=no_brackets)
    bracket_label: BracketStyle = field(default_factory=no_brackets)
    bracket_year: BracketStyle = field(default_factory=no_brackets)


def setup(app):
    """Register the bibliography and in-text citation styles with Sphinx."""

    pybtex.plugin.register_plugin("pybtex.style.formatting", "apa", APAStyle)
    sphinxcontrib.bibtex.plugin.register_plugin(
        "sphinxcontrib.bibtex.style.referencing",
        "author_year_no_brackets",
        AuthorYearNoBrackets,
    )

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
