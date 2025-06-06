import os

import requests
from tqdm import tqdm


def download_file(url, dest_path, quiet=False, overwrite=False):
    """
    Download a file from a given URL to a specified destination directory.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    dest_path : str
        The destination directory or filename where the file will be saved.
        If a directory, it should already exist.  If a filename, its directory
        will be created if it does not exist.
    quiet : bool, optional
        If True, suppress the progress bar. Default is False.
    overwrite : bool, optional
        If True, overwrite the file if it already exists. Default is False.

    Examples
    --------
    >>> download_file('http://example.com/file1.txt', '/path/to/destination', quiet=True, overwrite=True)
    """  # noqa: E501
    if os.path.isdir(dest_path):
        dest_dir = dest_path
        dest_filename = os.path.join(dest_dir, url.split('/')[-1])
    else:
        dest_dir = os.path.dirname(dest_path)
        dest_filename = dest_path
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    if not overwrite and os.path.exists(dest_filename):
        if not quiet:
            print(f'File {dest_filename} already exists, skipping download.')
        return
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(dest_filename, 'wb') as file:
        if quiet:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
        else:
            with tqdm(
                desc=dest_filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
    if not quiet:
        print(f'Downloaded {dest_filename}')


def download_files(files, base_url, dest_dir, quiet=False, overwrite=False):
    """
    Download multiple files from a base URL to a specified destination
    directory.

    Parameters
    ----------
    files : list of str
        The list of file paths to download from the base URL.
    base_url : str
        The base URL from which the files will be downloaded.
    dest_dir : str
        The destination directory where the files will be saved.
    quiet : bool, optional
        If True, suppress the progress bar. Default is False.
    overwrite : bool, optional
        If True, overwrite the files if they already exist. Default is False.

    Examples
    --------
    >>> download_files(['file1.txt', 'subdir/file2.txt'], 'http://example.com/', '/path/to/destination', quiet=True, overwrite=True)
    """  # noqa: E501
    for file_path in files:
        url = os.path.join(base_url, file_path)
        dest_path = os.path.join(dest_dir, file_path)
        download_file(url, dest_path, quiet, overwrite)
