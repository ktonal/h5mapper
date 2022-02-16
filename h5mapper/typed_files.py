from time import time
import click
import h5mapper as h5m

__all__ = [
    'SoundBank',
    'sound_bank',
    'image_bank'
]


class SoundBank(h5m.TypedFile):
    snd: h5m.Sound = None
    file_labels: h5m.FilesLabels = None
    dir_labels: h5m.DirLabels = None


@click.command()
@click.option("-t", "--target", help="file to be created")
@click.option("-s", "--source", help="where to search for sound files")
@click.option("--sr", "-r", default=22050, help="the sample rate used for loading the files")
@click.option("--mono", "-m", default=True, help="whether to force conversion to mono")
@click.option("--normalize", "-n", default=True, help="whether each file should be normalized")
@click.option("--file-labels", "-f", default=False, help="whether file labels should be added")
@click.option("--dir-labels", "-d", default=False, help="whether directory labels should be added")
@click.option("--parallelism", "-p", default='mp', help="flavor of parallelism to use."
                                                        " Must be one of ['mp', 'threads', 'none']")
@click.option("--n-workers", "-w", default=8, help="number of workers to use")
def sound_bank(target, source,
               sr=22050,
               mono=True,
               normalize=True,
               file_labels=False,
               dir_labels=False,
               parallelism='mp',
               n_workers=8):
    source = [source] if isinstance(source, str) else source
    # get all the files under `source` with an image extension
    files = list(h5m.FileWalker(h5m.Sound.__re__, source))
    N = len(files)
    click.echo(f"consolidating {N} files into '{target}'...")
    start = time()
    # dynamically create the TypedFile
    SoundBank.snd = h5m.Sound(sr=sr, mono=mono, normalize=normalize)
    if file_labels:
        SoundBank.file_labels = h5m.FilesLabels(derived_from='snd')
    if dir_labels:
        SoundBank.dir_labels = h5m.DirLabels()
    # parallel job
    h5f = SoundBank.create(target, files, parallelism=parallelism, n_workers=n_workers)
    dur = time() - start
    click.echo(f"stored {N} files in {'%.3f' % dur} seconds")
    # echo
    h5f.info()
    h5f.close()
    return h5f


@click.command()
@click.option("-t", "--target", help="file to be created")
@click.option("-s", "--source", help="where to search for images")
@click.option("--no-vshape", "-v", is_flag=True, help="whether the source images have different shapes")
@click.option("--file-labels", "-f", default=False, help="whether file labels should be added")
@click.option("--dir-labels", "-d", default=False, help="whether directory labels should be added")
@click.option("--parallelism", '-p', default='mp', help="flavor of parallelism to use."
                                                        " Must be one of ['mp', 'threads', 'none']")
@click.option("--n-workers", "-w", default=8, help="number of workers to use")
def image_bank(target, source,
               no_vshape=False,
               file_labels=False,
               dir_labels=False,
               parallelism='mp',
               n_workers=8):
    # get all the files under `source` with an image extension
    files = list(h5m.FileWalker(h5m.Image.__re__, [source]))
    N = len(files)
    click.echo(f"consolidating {N} files into '{target}'...")
    start = time()
    # dynamically create a TypedFile
    ftp = h5m.typedfile('ImageBank', dict(
        img=h5m.Image() if no_vshape else h5m.VShape(h5m.Image()),
        **({"file_labels": h5m.FilesLabels(derived_from='img')} if file_labels else {}),
        **({"dir_labels": h5m.DirLabels()} if dir_labels else {}),
    ))
    # parallel job
    h5f = ftp.create(target, files, parallelism=parallelism, n_workers=n_workers)
    # echo
    dur = time() - start
    click.echo(f"stored {N} files in {'%.3f' % dur} seconds")
    h5f.info()
    h5f.close()
    return h5f
