import click

@click.group()
def KernelPheno():
    """ Kernel Vitreousness project management and phenotyping tools """
    pass


from zooexp import *
from preprocess import *


if __name__ == '__main__':
    KernelPheno()
