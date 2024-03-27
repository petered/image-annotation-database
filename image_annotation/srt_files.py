import re
from datetime import datetime
from typing import Sequence, Optional, Tuple
import pysrt

from artemis.general.utils_for_testing import hold_tempfile
from artemis.image_processing.video_frame import FrameGeoData
from image_annotation.kml_files import write_kml_file


def read_dji_srt_file(srt_path: str) -> Sequence[FrameGeoData]:
    subs = pysrt.open(srt_path)
    items = []
    for i, item in enumerate(subs):

        new_format_match = re.search(r"GPS \(([-\w\.]+), ([-\w\.]+), ([-\w\.]+)\)", item.text_without_tags.strip('\n'))
        if new_format_match:
            long, lat, alt = new_format_match.groups()
            geodata = FrameGeoData(
                lat_long=(float(lat), float(long)),
                altitude_from_home=float(alt),
                epoch_time_us=int(item.start.ordinal*1000)
                )
            items.append(geodata)
        else:
            dtime_info, abstime_info, field_string = item.text_without_tags.strip('\n').split('\n')
            parsed_dict = {}
            for match in re.finditer(r"\[(\w+) *: *([-\w\.]+)\]", field_string):
                key, value = match.groups()
                parsed_dict[key.strip()] = value
            if abstime_info.count(',') == 2:  # Example format: '2023-02-26 08:12:41,410,834'
                dtime, millis, micros = abstime_info.split(',')
                dtime_to_second = datetime.fromisoformat(dtime)
                timestamp_us = int(dtime_to_second.timestamp()*1000000)+int(millis)*1000+int(micros)
            elif abstime_info.count(',')==0 and abstime_info.count('.')==1:  # Example format: '2023-07-06 11:08:24.659'
                dtime, millis = abstime_info.split('.')
                dtime_to_second = datetime.fromisoformat(dtime)
                timestamp_us = int(dtime_to_second.timestamp()*1000000)+int(millis)*1000
            else:
                raise Exception(f"Unrecognized timestamp format: {abstime_info}")
                # Example format: '2023-07-06 11:08:24.659'
            lat = float(parsed_dict['latitude'])
            long = float(parsed_dict.get('longitude', parsed_dict.get('longtitude')))  # Yes DJI made a typo but then fixed it later.
            items.append(FrameGeoData(
                lat_long=(lat, long) if lat != 0 and long != 0 else None,  # Hope you're not flying off the west coast of Africa
                altitude_from_home=float(parsed_dict['altitude']) if 'altitude' in parsed_dict else None,
                epoch_time_us=timestamp_us
            ))
    return items


def cut_srt_file(srt_path: str, new_srt_path: str, time_interval: Tuple[Optional[float], Optional[float]]):
    # subs = pysrt.open(srt_path)
    # items = []
    # with hold_tempfile(ext='.srt') as temp_srt_path:
    #     for i, item in enumerate(subs):
    #         new_format_match = re.search(r"GPS \(([-\w\.]+), ([-\w\.]+), ([-\w\.]+)\)", item.text_without_tags.strip('\n'))
    #         if new_format_match:
    #             epoch_time_s = item.start.ordinal/1000
    #             if time_interval[0] is not None and epoch_time_s < time_interval[0]:
    #                 continue
    #             if time_interval[1] is not None and epoch_time_s > time_interval[1]:
    #                 break
    #             items.append(item)

    # Now save the existing items
    pysrt.open(srt_path).slice(starts_after={'seconds': time_interval[0]}, ends_after={'seconds': time_interval[1]}).save(new_srt_path)

def srt_to_kml(srt_path: str, kml_path: str, kml_object_name: str):
    frame_data = read_dji_srt_file(srt_path)
    latlong_pairs = [f.lat_long for f in frame_data]
    write_kml_file(path=kml_path, name=kml_object_name, latlng_coords=latlong_pairs)


if __name__ == "__main__":
    srt_to_kml(
        srt_path='/Users/peter/drone/e2_dual/raw/dji_2023-05-03_00-37-54_0316.srt',
        kml_path='~/Downloads/Flight1.kml',
        kml_object_name='Flight1'
    )
    srt_to_kml(
        srt_path='/Users/peter/drone/e2_dual/raw/dji_2023-05-03_01-07-28_0318.srt',
        kml_path='~/Downloads/Flight2.kml',
        kml_object_name='Flight2'
    )
    srt_to_kml(
        srt_path='/Users/peter/drone/dji_air2s/raw/dji_2023-05-03_01-47-34_0777.srt',
        kml_path='~/Downloads/Flight4.kml',
        kml_object_name='Flight4'
    )
    srt_to_kml(
        srt_path='/Users/peter/drone/dji_air2s/raw/dji_2023-05-03_01-49-36_0778.srt',
        kml_path='~/Downloads/Flight5.kml',
        kml_object_name='Flight5'
    )