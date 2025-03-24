import os
import zipfile

from i7aof.download import download_file


def download_imbie():
    """
    Download the geojson files that define the IMBIE basins
    """

    sample_output = 'imbie/ANT_Basins_IMBIE2_v1.6/ANT_Basins_IMBIE2_v1.6.shp'
    if os.path.exists(sample_output):
        return

    filename = 'ANT_Basins_IMBIE2_v1.6.zip'
    dest_path = os.path.join('imbie', filename)
    url = f'http://imbie.org/wp-content/uploads/2016/09/{filename}'

    download_file(url=url, dest_path=dest_path)

    print('Decompressing IMBIE2 data...')
    # unzip
    with zipfile.ZipFile('imbie/ANT_Basins_IMBIE2_v1.6.zip', 'r') as f:
        f.extractall('imbie/ANT_Basins_IMBIE2_v1.6/')
    print('  Done.')
