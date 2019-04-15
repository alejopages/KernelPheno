from main import KernelPheno

import click
import os.path as osp
import subprocess as sp

@click.command()
@click.argument(
    'dir'
)
@click.option(
    '-f',
    'format',
    help="Format to convert the images to",
    default='jpg',
    show_default=True
)
def convert(dir, format):

    if not osp.isdir(dir):
        print("This directory does not exist: " + dir)
        exit()

    mog_proc = sp.run(['mogrify', '-format', format, osp.join(dir, '*')])
    return

KernelPheno.add_command(convert)


if __name__ == '__main__':
    cnvjpg(sys.argv[1])
