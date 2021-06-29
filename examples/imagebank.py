import h5m
import click
import os
from time import time

"""
CLI for an Image Bank
"""


@click.command()
@click.option("-t", "--target", help="file to be created")
@click.option("-s", "--source", help="where to search for images")
@click.option("--no-vshape", is_flag=True, help="whether the source images have different shapes")
@click.option("--parallelism", default='mp', help="flavor of parallelism to use."
                                                  " Must be one of ['mp', 'future', 'none']")
@click.option("--n-workers", "-w", default=8, help="number of workers to use")
def main(target, source, no_vshape=False, parallelism='mp', n_workers=8):
    # get all the files under `source` with an image extension
    files = list(h5m.FileWalker(h5m.Image.__re__, [source]))
    N = len(files)
    click.echo(f"consolidating {N} files into '{target}'...")
    start = time()
    # dynamically create a FileType
    ftp = h5m.filetype('ImageBank', dict(
        img=h5m.Image() if no_vshape else h5m.VShape(h5m.Image())
    ))
    # parallel job
    h5f = ftp.create(target, files, parallelism=parallelism, n_workers=n_workers)
    # echo
    h5f.info()
    dur = time() - start
    click.echo(f"stored {N} files in {'%.3f' % dur} seconds")

    # it's just a demo...
    os.remove(target)


if __name__ == '__main__':
    # invoke with `python path/to/imagebank_cli.py [OPTIONS] `
    # or add main() as entry-point to your setup.py
    target = main()
