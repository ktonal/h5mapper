import os
import re
from typing import Iterable

__all__ = [
    "FileWalker"
]


class FileWalker:

    def __init__(self, regex, sources=None):
        """
        recursively find files from `sources` whose paths match the pattern passed in `regex`

        Parameters
        ----------
        regex : str or re.Pattern
            the pattern a path must match
        sources : str or iterable of str
            a single path (string, os.Path) or an iterable of paths.
            Each item can either be the path to a single file or to a directory,
            in which case, it will be walked recursively.

        Examples
        --------
        >>> files = list(FileWalker(regex=r'.*mid$', sources=["my-root-dir", 'piece.mid']))

        """
        self._regex = re.compile(regex)
        self.sources = sources

    def __iter__(self):
        generators = []

        if self.sources is not None and isinstance(self.sources, Iterable):
            if isinstance(self.sources, str):
                if not os.path.exists(self.sources):
                    raise FileNotFoundError("%s does not exist." % self.sources)
                if os.path.isdir(self.sources):
                    generators += [self.walk_root(self.sources)]
                else:
                    if self.is_matching_file(self.sources):
                        generators += [[self.sources]]
            else:
                for item in self.sources:
                    if not os.path.exists(item):
                        raise FileNotFoundError("%s does not exist." % item)
                    if os.path.isdir(item):
                        generators += [self.walk_root(item)]
                    else:
                        if self.is_matching_file(item):
                            generators += [[item]]

        for generator in generators:
            for file in generator:
                yield file

    def walk_root(self, root):
        for directory, _, files in os.walk(root):
            for file in filter(self.is_matching_file,
                               (os.path.join(directory, f) for f in files)):
                yield file

    def is_matching_file(self, filename):
        # filter out any hidden files
        if os.path.split(filename.strip("/"))[-1].startswith("."):
            return False
        return bool(re.search(self._regex, filename))
