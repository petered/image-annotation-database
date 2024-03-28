import os
from dataclasses import replace

import cv2
import numpy as np
from tinydb import Query

from image_annotation.annotated_image_serialization import load_tiff_metadata
from image_annotation.annotation_database import AnnotationDatabaseAccessor, Annotation, FrameSourceInfo, FrameSourceInfoAndImage
from image_annotation.goto_queries import RecordQuery
from image_annotation.helpers_for_testing import asset_path_for_testing
from image_annotation.srt_files import read_dji_srt_file
from artemis.general.utils_for_testing import hold_tempdir
from artemis.image_processing.media_metadata import read_image_geodata_or_none
from artemis.image_processing.video_segment import VideoSegment
from image_annotation.geo_utils import haversine_distance


def load_test_fsi_image_from_image_file() -> FrameSourceInfoAndImage:

    source_info = FrameSourceInfo(
        source_file=asset_path_for_testing('data/basalt_canyon.jpg'),
        record_query=RecordQuery(nickname='test_collection'),
        annotations=[Annotation(ijhw_box=(966, 1210, 38, 32), label='people', value=1, description=''),
                     Annotation(ijhw_box=(498, 1555, 42, 44), label='cliff_tree', value=1, description='')],
        geodata=read_image_geodata_or_none(asset_path_for_testing('data/basalt_canyon.jpg'))
    )
    image = cv2.imread(source_info.source_file)
    return FrameSourceInfoAndImage(source_info, image)


def load_test_fsi_image_from_video_frame() -> FrameSourceInfoAndImage:
    # goto:file=/Users/peter/projects/eagle_eyes_video_scanner/dist/EagleEyesScan.app/Contents/Frameworks/assets/data/cutblock.mov&index=21&ij=(872, 3704)&zoom=2.0
    frame_index = 21
    source = VideoSegment(asset_path_for_testing('data/cutblock.mov'))
    source_info = FrameSourceInfo(
        source_file=source.path,
        source_index=frame_index,
        record_query=RecordQuery(nickname='test_collection'),
        annotations=[
            Annotation(ijhw_box=(872, 3704, 100, 100), label='orange', value=1, description='orange guy behind bush')
        ],
        geodata=read_dji_srt_file(asset_path_for_testing('data/cutblock.srt'))[frame_index]
    )
    image = source.get_reader().request_frame(frame_index).image
    return FrameSourceInfoAndImage(source_info, image)


def test_annotation_db_access():
    with hold_tempdir() as db_path:
        db = AnnotationDatabaseAccessor(
            db_path,
            source_data_base_path=None,
        )

        fsii = load_test_fsi_image_from_image_file()

        # source_info = FrameSourceInfo(
        #     source_file=AssetImages.BASALT_CANYON,
        #     record_query=RecordQuery(nickname='test_collection'),
        #     annotations=[Annotation(ijhw_box=(966, 1210, 38, 32), label='people', value=1, description=''),
        #                  Annotation(ijhw_box=(498, 1555, 42, 44), label='cliff_tree', value=1, description='')]
        # )

        source_info = fsii.frame_source_info
        source_id = source_info.get_source_identifier()

        image = cv2.imread(asset_path_for_testing('data/basalt_canyon.jpg'))
        assert image.shape == (1500, 2000, 3)
        assert db.lookup_frame_source_info_from_identifier(source_id) is None
        assert db.query_annotation_data() == []

        db.save_annotated_image(frame_source_info=source_info, image=image, save_image=True, save_thumbnails=True)

        # Empty query should return all annotations
        retrieved_source_infos = db.query_annotation_data()
        assert len(retrieved_source_infos) == 1
        assert retrieved_source_infos[0] == source_info
        assert db.lookup_frame_source_info_from_identifier(source_id) == source_info

        # Query with no match
        retrieved_source_infos = db.query_annotation_data(Query().record_query.nickname == 'non-existent')
        assert len(retrieved_source_infos) == 0

        # Query by source identifier
        retrieved_source_infos = db.query_annotation_data(Query().record_query.nickname == 'test_collection')
        assert len(retrieved_source_infos) == 1
        assert retrieved_source_infos[0] == source_info

        # Query by file paths
        # multipath: str = ';'.join([AssetImages.HEILO_AZUL_360, AssetImages.BASALT_CANYON, AssetImages.ABYSS_2])
        multipath = ';'.join([asset_path_for_testing('data/heilo_azul_360.jpg'), asset_path_for_testing('data/basalt_canyon.jpg'), asset_path_for_testing('data/abyss_road/dji_2022-08-30_16-09-03_0516.jpg')])
        retrieved_index_info_pairs = db.query_annotation_data_from_path(multipath)
        assert len(retrieved_index_info_pairs) == 1
        assert retrieved_index_info_pairs[0] == replace(source_info, source_file=multipath, source_index=1)

        retrieved_annotated_image = db.load_annotated_image(source_id)
        assert retrieved_annotated_image is not None
        assert retrieved_annotated_image.annotations == source_info.annotations
        assert np.array_equal(retrieved_annotated_image.image, image)
        assert db.load_annotated_image('non-identifier') is None

        db.delete_annotation_data(source_id)
        assert db.lookup_frame_source_info_from_identifier(source_id) is None


def test_serialization():

    image_fsii = load_test_fsi_image_from_image_file()
    video_fsii = load_test_fsi_image_from_video_frame()

    for frame_type, fsii in [('image', image_fsii), ('video', video_fsii)]:
        print(f"Saving and loading {frame_type}")

        with hold_tempdir() as tempdir:

            image_path = fsii.save_to_image_file(tempdir)

            n_values_in_image = fsii.image.size
            print(f"Saved annotation path: {image_path}")
            if frame_type == 'image':
                assert image_path.endswith('.jpg'), image_path
                compression_ratio = os.path.getsize(image_path) / n_values_in_image
                assert compression_ratio < 0.25
            else:
                assert image_path.endswith('.tiff'), image_path
                file_size = os.path.getsize(image_path)
                compression_ratio = file_size/n_values_in_image
                assert compression_ratio < 1.01, "It definitely shouldn't be growing"
            print(f'Compression Ratio: {compression_ratio:.0%}')

            new_fsii = FrameSourceInfoAndImage.load_from_image_file(image_path)
            assert np.array_equal(fsii.image, new_fsii.image)
            assert fsii.frame_source_info == new_fsii.frame_source_info

            # Check that metadata is saved
            metadata = load_tiff_metadata(image_path)
            assert haversine_distance(fsii.frame_source_info.geodata.lat_long, (metadata.gps_info.latitude, metadata.gps_info.longitude)) < 10
            assert abs(metadata.date_time.timestamp() - fsii.frame_source_info.geodata.get_timestamp()) < 1
        print("Success")


def test_write_annotation_to_directory_with_nonlatin():

    with hold_tempdir() as tempdir:
        subdir = os.path.join(tempdir, '中文')
        db = AnnotationDatabaseAccessor(subdir, source_data_base_path=None)
        fsii = load_test_fsi_image_from_image_file()
        assert db.load_annotated_image(fsii.frame_source_info.get_source_identifier()) is None
        db.save_annotated_image(frame_source_info=fsii.frame_source_info, image=fsii.image, save_image=True, save_thumbnails=True)
        assert os.path.exists(subdir)
        fsii_reloaded = db.load_annotated_image(fsii.frame_source_info.get_source_identifier())
        assert fsii_reloaded is not None
        assert np.array_equal(fsii.image, fsii_reloaded.image)


if __name__ == '__main__':
    # test_annotation_db_access()
    # test_serialization()
    test_write_annotation_to_directory_with_nonlatin()