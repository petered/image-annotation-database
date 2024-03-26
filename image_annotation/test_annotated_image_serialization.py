import datetime

import numpy as np

from artemis.general.utils_for_testing import hold_tempfile
from image_annotation.annotated_image_serialization import save_tiff_with_metadata, load_tiff_with_metadata, TiffImageMetadata, GPSInfo


def test_save_load_image_with_metadata():

    with hold_tempfile(ext='.tiff') as test_file_path:

        # Test data: Create a simple blue-green gradient image array
        image_array = np.zeros((100, 100, 3), dtype=np.uint8)
        image_array[:, :, 1] = np.linspace(0, 255, 100)  # Green gradient
        image_array[:, :, 0] = np.linspace(255, 0, 100)  # Blue gradient

        # Test metadata
        json_metadata = {
            "description": "Test blue-green gradient",
            "author": "Test Author",
            "version": "1.0",
            "box_1" : [200, 300, 400, 500]
        }

        metadata = TiffImageMetadata(
            # 4pm Eastern time on 2021-10-01
            date_time=datetime.datetime(2021, 10, 1, 16, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=-4))),
            gps_info=GPSInfo(latitude=37.7749, longitude=-122.4194, altitude=476.433),
            jsonable_metadata=json_metadata
        )

        # Save the image and metadata
        save_tiff_with_metadata(image_array, test_file_path, metadata=metadata)

        # Load the image and metadata back
        loaded_image_array, loaded_metadata = load_tiff_with_metadata(test_file_path)

        # Test assertions
        assert np.array_equal(image_array, loaded_image_array), "Loaded image array does not match the original."
        assert json_metadata == loaded_metadata.jsonable_metadata, "Loaded metadata does not match the original."

        assert abs(metadata.gps_info.latitude - loaded_metadata.gps_info.latitude) < 1e-4, f"Loaded latitude does not match the original. {metadata.gps_info.latitude} != {loaded_metadata.gps_info.latitude}"
        assert abs(metadata.gps_info.longitude - loaded_metadata.gps_info.longitude) < 1e-4, f"Loaded longitude does not match the original. {metadata.gps_info.longitude} != {loaded_metadata.gps_info.longitude}"
        assert abs(metadata.gps_info.altitude - loaded_metadata.gps_info.altitude) < 1e-2, f"Loaded altitude does not match the original. {metadata.gps_info.altitude} != {loaded_metadata.gps_info.altitude}"

        assert metadata.date_time == loaded_metadata.date_time, f"Loaded date time does not match the original. {metadata.date_time} != {loaded_metadata.date_time}"
        print("Test passed successfully!")


if __name__ == "__main__":
    test_save_load_image_with_metadata()
