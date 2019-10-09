import urllib.request
from os import listdir, makedirs
from os.path import isdir, isfile, join, exists, dirname


def download_file(url, archive_filename):
    def progress(blocknum, bs, size):
        total_sz_mb = '%.2f MB' % (size / 1e6)
        current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
        print('\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb), end='')
    print("Downloading %s" % url)
    urllib.request.urlretrieve(url, filename=archive_filename, reporthook=progress)
    print("")


def download_file_if_not_exists(url, archive_path):
    if exists(archive_path): return
    create_if_not_exist(dirname(archive_path))
    download_file(url,archive_path)


def ls(dir, typecheck):
    el = [f for f in listdir(dir) if typecheck(join(dir, f))]
    el.sort()
    return el


def list_dirs(dir):
    return ls(dir, typecheck=isdir)


def list_files(dir):
    return ls(dir, typecheck=isfile)


def create_if_not_exist(path):
    if not exists(path): makedirs(path)

