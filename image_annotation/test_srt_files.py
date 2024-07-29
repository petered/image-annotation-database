import os

from artemis.general.debug_utils import easy_profile
from artemis.general.utils_for_testing import hold_tempdir
from artemis.image_processing.media_metadata import read_image_geodata_or_none
from image_annotation.file_utils import DroneDataDirectory
from image_annotation.helpers_for_testing import asset_path_for_testing
from image_annotation.srt_files import read_dji_srt_file, srt_to_kml


def test_srt_to_lat_long():

    # srt_path = AssetFiles.CUTBLOCK_SRT
    srt_path = asset_path_for_testing('data/cutblock.srt')
    items =read_dji_srt_file(srt_path)

    srt_path = DroneDataDirectory().get_file('test_data/misc/DJI_0289.SRT')
    items =read_dji_srt_file(srt_path)
    assert 40 < items[0].lat_long[0] < 60
    assert -110 < items[0].lat_long[1] < 30
    dt = items[0].get_datetime()
    assert dt.year == 2024
    assert dt.month == 7
    assert dt.hour == 19

    srt_path = DroneDataDirectory().get_file('test_data/misc/dji_2023-07-06_18-07-50_0005.srt')
    items = read_dji_srt_file(srt_path)
    assert 49 < items[0].lat_long[0] < 51
    assert -124 < items[0].lat_long[1] < -123
    dt = items[0].get_datetime()
    assert dt.year == 2023
    assert dt.month == 7
    assert dt.tzname() == "PDT"

    srt_path = DroneDataDirectory().get_file('test_data/misc/dji_2023-02-26_15-12-52_0168.srt')
    items = read_dji_srt_file(srt_path)
    assert 37 < items[0].lat_long[0] < 39
    assert -124 < items[0].lat_long[1] < -123
    dt = items[0].get_datetime()
    assert dt.year == 2023
    assert dt.month == 2
    assert dt.tzname() == "PST"

    items = read_dji_srt_file(DroneDataDirectory().get_file('test_data/misc/dji_2023-05-03_00-37-54_0316.srt'))
    assert ((dt:=items[0].get_datetime()).year, dt.month, dt.day)==(2023, 5, 2)
    items = read_dji_srt_file(DroneDataDirectory().get_file('test_data/misc/dji_2023-05-03_01-47-34_0777.srt'))
    ((dt := items[0].get_datetime()).year, dt.month, dt.day) == (2023, 5, 2)


def test_get_exif_from_image():

    image_path = asset_path_for_testing('data/basalt_canyon.jpg')
    with easy_profile(f"Reading exif data from image: {image_path}"):
        geodata = read_image_geodata_or_none(image_path)
    assert geodata.get_latlng_str() == '50.07795, -123.31487'
    image_path = asset_path_for_testing('eagle_eyes_mosaic_help.png')
    with easy_profile(f"Reading exif data from image: {image_path}"):
        geodata = read_image_geodata_or_none(image_path)
    assert geodata is None


def test_srt_to_kml():

    with hold_tempdir() as fdir:

        srt_to_kml(
            srt_path=DroneDataDirectory().get_file('test_data/misc/dji_2023-05-03_00-37-54_0316.srt'),
            kml_path=os.path.join(fdir, 'Flight1.kml'),
            kml_object_name='Flight1'
        )
        # srt_to_kml(
        #     srt_path=DroneDataDirectory().get_file('e2_dual/raw/dji_2023-05-03_01-07-28_0318.srt'),
        #     kml_path=os.path.join(fdir, 'Flight2.kml'),
        #     kml_object_name='Flight2'
        # )
        # srt_to_kml(
        #     srt_path=DroneDataDirectory().get_file('dji_air2s/raw/dji_2023-05-03_01-47-34_0777.srt'),
        #     kml_path=os.path.join(fdir, 'Flight4.kml'),
        #     kml_object_name='Flight4'
        # )
        srt_to_kml(
            srt_path=DroneDataDirectory().get_file('test_data/misc/dji_2023-05-03_01-49-36_0778.srt'),
            kml_path=os.path.join(fdir, 'Flight5.kml'),
            kml_object_name='Flight5'
        )


if __name__ == "__main__":
    test_srt_to_lat_long()
    test_get_exif_from_image()
    test_srt_to_kml()
