import os
import re
from typing import Iterable

__all__ = [
    "EXTENSIONS",
    "FileWalker"
]

AUDIO_EXTENSIONS = r"[wav$|aif$|aiff$|mp3$|m4a$|mp4$]"
IMAGE_EXTENSIONS = r"[png$|jpeg$]"
MIDI_EXTENSIONS = r"mid$"

EXTENSIONS = dict(audio=AUDIO_EXTENSIONS,
                  img=IMAGE_EXTENSIONS,
                  midi=MIDI_EXTENSIONS,
                  none={})


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
        if isinstance(regex, str):
            regex = [regex]
        self._regex = re.compile(regex)

        generators = []

        if sources is not None and isinstance(sources, Iterable):
            if isinstance(sources, str):
                if not os.path.exists(sources):
                    raise FileNotFoundError("%s does not exist." % sources)
                if os.path.isdir(sources):
                    generators += [self.walk_root(sources)]
                else:
                    if self.is_matching_file(sources):
                        generators += [[sources]]
            else:
                for item in sources:
                    if not os.path.exists(item):
                        raise FileNotFoundError("%s does not exist." % item)
                    if os.path.isdir(item):
                        generators += [self.walk_root(item)]
                    else:
                        if self.is_matching_file(item):
                            generators += [[item]]

        self.generators = generators

    def __iter__(self):
        for generator in self.generators:
            for file in generator:
                yield file

    def walk_root(self, root):
        for directory, _, files in os.walk(root):
            for file in filter(self.is_matching_file, files):
                yield os.path.join(directory, file)

    def is_matching_file(self, filename):
        # filter out any hidden files
        if os.path.split(filename.strip("/"))[-1].startswith("."):
            return False
        return bool(re.match(self._regex, filename))
