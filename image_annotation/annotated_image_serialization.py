import traceback
from dataclasses import dataclass, field
from datetime import datetime, tzinfo, timezone
from typing import Any, Mapping, Tuple, Optional, TypedDict
import cv2
from PIL import Image
import numpy as np
import json
import piexif
from piexif import GPSIFD
from piexif.helper import UserComment


@dataclass
class GPSInfo:
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None


@dataclass
class TiffImageMetadata:
    """ Some standardized metadata that can be stored in a TIFF file
    Note that because of constraints on the structure of EXIF data,
    the precise values of date_time and gps_info may change slightly
    when saved and loaded back from a TIFF file.

    The data in jsonable_metadata will be identical though.
    """
    date_time: Optional[datetime] = None  # Date time (with timezone) of the image
    gps_info: Optional[GPSInfo] = None
    jsonable_metadata: Optional[Mapping[str, Any]] = field(default=None)


def numdem_to_float(num_dem: Tuple[int, int]) -> float:
    return num_dem[0] / num_dem[1]


def decimal_degree_to_dms_num_dem(value: float, loc: str) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]], str]:
    """Convert decimal degrees to degrees, minutes, seconds tuple in EXIF format."""
    if value < 0:
        loc_value = loc[1]
    else:
        loc_value = loc[0]
    abs_value = abs(value)
    deg = int(abs_value)
    min = int((abs_value - deg) * 60)
    sec = (abs_value - deg - min / 60) * 3600 * 100

    # Format for EXIF
    deg = (deg, 1)
    min = (min, 1)
    sec = (int(sec), 100)
    return (deg, min, sec), loc_value


def dms_num_dem_to_decimal_degree(dms: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]], loc: str) -> float:
    """Convert degrees, minutes, seconds tuple in EXIF format to decimal degrees."""
    # deg, min, sec = dms
    # value = deg[0] + min[0]/60 + sec[0]/3600
    # (deg_num, deg_denom), (min_num, min_denom), (sec_num, sec_denom) = dms
    deg_numdem, min_numdem, sec_numdem = dms
    value = numdem_to_float(deg_numdem) + numdem_to_float(min_numdem) / 60 + numdem_to_float(sec_numdem) / 3600
    if loc in ['S', 'W']:
        value = -value
    else:
        assert loc in ['N', 'E'], f"Invalid loc: {loc}"
    return value


def metadata_to_exif_dict(metadata: TiffImageMetadata) -> Mapping[str, Any]:
    """ Turn a metadata object into exif bytes in the standard format """
    exif_dict = {"GPS": {}, "Exif": {}}

    # Serialize jsonable metadata to a JSON string and include it in EXIF
    if metadata.jsonable_metadata:
        json_metadata = json.dumps(metadata.jsonable_metadata)
        exif_dict['Exif'][piexif.ExifIFD.UserComment] = UserComment.dump(json_metadata)

    # DateTime
    date_time = metadata.date_time
    if metadata.date_time:
        # Store utc time in 0th and local time in Exif
        dt_in_utc = date_time.astimezone(timezone.utc)
        exif_dict['0th'] = {piexif.ImageIFD.DateTime: dt_in_utc.strftime("%Y:%m:%d %H:%M:%S")}
        exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = date_time.strftime("%Y:%m:%d %H:%M:%S")

        #
        # metadata_dict['0th'] = {piexif.ImageIFD.DateTime: metadata.date_time.strftime("%Y:%m:%d %H:%M:%S") if metadata.date_time else ''}
        # metadata_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = metadata.date_time.strftime("%Y:%m:%d %H:%M:%S") if metadata.date_time else ''

    # GPS Information
    if metadata.gps_info:
        if metadata.gps_info.latitude is not None and metadata.gps_info.longitude is not None:
            exif_dict['GPS'][piexif.GPSIFD.GPSLatitude], exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] \
                = decimal_degree_to_dms_num_dem(metadata.gps_info.latitude, "NS")
            exif_dict['GPS'][piexif.GPSIFD.GPSLongitude], exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] \
                = decimal_degree_to_dms_num_dem(metadata.gps_info.longitude, "EW")
            # metadata_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = 'E' if metadata.gps_info.longitude >= 0 else 'W'
            # metadata_dict['GPS'][piexif.GPSIFD.GPSLongitude] = decimal_degree_to_dms_num_dem(metadata.gps_info.longitude, "EW")

        if metadata.gps_info.altitude is not None:
            exif_dict['GPS'][piexif.GPSIFD.GPSAltitudeRef] = 0 if metadata.gps_info.altitude >= 0 else 1
            exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (abs(int(metadata.gps_info.altitude * 100)), 100)

    return exif_dict


def copy_image_and_patch_on_metadata(source_path: str, dest_path: str, metadata: TiffImageMetadata):

    exif_dict = metadata_to_exif_dict(metadata)
    # TODO: Losslessly over the image to the new path - attaching the exif data.

    # Convert the EXIF dictionary to bytes
    exif_bytes = piexif.dump(exif_dict)

    # Open the source image, insert the new EXIF data, and save to dest_path
    # This operation is done without re-encoding the JPEG image data
    piexif.insert(exif_bytes, source_path, dest_path)


def save_tiff_with_metadata(image_array: np.ndarray, path: str, metadata: TiffImageMetadata):
    """Save an image array as a TIFF file with embedded JSON serialized metadata."""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)

    exif_bytes = piexif.dump(metadata_to_exif_dict(metadata))

    # Save the image with EXIF data
    image.save(path, "TIFF", exif=exif_bytes)
    # Note - we tried lossless compression
    # compression='tiff_lzw' - Fails to save with exif, and actually grows file by 18%.
    # compression='tiff_deflate' - Shrinks file by 14% but still fails to save with exif.
    # Not worth it - so we just save raw.


def load_tiff_metadata(path: str) -> TiffImageMetadata:
    """Load JSON serialized metadata from a TIFF file."""
    # Extract EXIF data
    exif_data = piexif.load(path)

    # Deserialize JSON metadata from custom EXIF tag if exists
    if 'Exif' in exif_data and piexif.ExifIFD.UserComment in exif_data['Exif']:
        metadata_int_tuple = exif_data['Exif'][piexif.ExifIFD.UserComment]
        json_metadata_str = piexif.helper.UserComment.load(bytes(metadata_int_tuple))
        json_metadata = json.loads(json_metadata_str)
    else:
        json_metadata = None

    # Convert EXIF data back to ExifDataDict format
    dt_entry = exif_data.get('0th', {}).get(piexif.ImageIFD.DateTime, None)
    if dt_entry:
        datetime_local = datetime.strptime(dt_entry.decode('utf-8'), "%Y:%m:%d %H:%M:%S")
        datetime_utc = datetime.strptime(exif_data.get('0th', {}).get(piexif.ImageIFD.DateTime, '').decode('utf-8'), "%Y:%m:%d %H:%M:%S")
        tz_offset = datetime_local - datetime_utc
        datetime_localized = datetime_local.replace(tzinfo=timezone(tz_offset))
    else:
        datetime_localized = None

    gps_entry = exif_data.get('GPS', {})
    if gps_entry:
        try:
            gps_info = GPSInfo(
                latitude=dms_num_dem_to_decimal_degree(exif_data.get('GPS', {}).get(GPSIFD.GPSLatitude, ((0, 1), (0, 1), (0, 1))),
                                                       exif_data.get('GPS', {}).get(GPSIFD.GPSLatitudeRef, '').decode('utf-8')),
                longitude=dms_num_dem_to_decimal_degree(exif_data.get('GPS', {}).get(GPSIFD.GPSLongitude, ((0, 1), (0, 1), (0, 1))),
                                                        exif_data.get('GPS', {}).get(GPSIFD.GPSLongitudeRef, '').decode('utf-8')),
                altitude=numdem_to_float(exif_data.get('GPS', {}).get(GPSIFD.GPSAltitude, (0, 1)))

            )
        except Exception as err:
            print(f"Error when attempting to read GPS data from {path}.  GPS info will be missing")
            print(traceback.format_exc())
            gps_info = None

    else:
        gps_info = None

    metadata = TiffImageMetadata(
        date_time=datetime_localized,
        gps_info=gps_info,
        jsonable_metadata=json_metadata,

    )

    return metadata


def load_tiff_with_metadata(path: str) -> Tuple[np.ndarray, TiffImageMetadata]:
    """Load an image array and its JSON serialized metadata from a TIFF file."""
    image = Image.open(path)
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    return image_bgr, load_tiff_metadata(path)
    #
    # # Extract EXIF data
    # # exif_data = piexif.load(image.info["exif"])
    # exif_data = piexif.load(path)
    #
    # # Deserialize JSON metadata from custom EXIF tag if exists
    # if 'Exif' in exif_data and piexif.ExifIFD.UserComment in exif_data['Exif']:
    #     metadata_int_tuple = exif_data['Exif'][piexif.ExifIFD.UserComment]
    #     # metadata_bytes = bytes(metadata_int_tuple)
    #     json_metadata_str = piexif.helper.UserComment.load(bytes(metadata_int_tuple))
    #     # json_metadata_str = piexif.helper.UserComment.load(exif_data['Exif'][piexif.ExifIFD.UserComment])
    #     json_metadata = json.loads(json_metadata_str)
    #     exif_data['jsonable_metadata'] = json_metadata
    #
    # # Convert EXIF data back to ExifDataDict format
    # metadata = ExifDataDict(DateTime=exif_data.get('DateTime', ''),
    #                         GPSInfo=exif_data.get('GPSInfo', {}),
    #                         jsonable_metadata=exif_data.get('jsonable_metadata', {}))
    #
    # return image_bgr, metadata
