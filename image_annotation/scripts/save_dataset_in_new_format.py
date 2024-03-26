import json
import os

from artemis.general.utils_for_testing import hold_tempdir
from artemis.image_processing.media_metadata import read_image_geodata_or_none
from image_annotation.annotation_database import FrameSourceInfoAndImage, FrameSourceInfo, Annotation
from image_annotation.basic_utils import is_image_path
from image_annotation.file_utils import create_ulid_for_file, get_associated_srt_file_or_none
from image_annotation.goto_queries import source_identifier_to_path_and_index, RecordQuery
from image_annotation.helpers_for_testing import asset_path_for_testing
from image_annotation.srt_files import read_dji_srt_file


def get_source_geodata_or_none(source_file: str, source_index: int):
    if is_image_path(source_file):
        return read_image_geodata_or_none(source_file)
    else:
        srt_file = get_associated_srt_file_or_none(source_file)
        if srt_file is not None:
            return read_dji_srt_file(srt_file)[source_index]


def save_dataset_in_new_format(
    original_dataset_folder = '/Volumes/WD_4TB/backup/eagle_eyes_dataset',
    new_dataset_folder = '/Volumes/WD_4TB/backup/eagle_eyes_dataset_v2',
    drone_source_data_folder = '/Volumes/WD_4TB/drone'
):

    json_file = os.path.join(original_dataset_folder, 'dataset.json')
    image_paths = [os.path.join(original_dataset_folder, f) for f in os.listdir(original_dataset_folder+'/images')]

    with hold_tempdir(path_if_successful=new_dataset_folder) as temp_new_dataset_folder:
        os.mkdir(os.path.join(temp_new_dataset_folder, 'images') )

        with open(json_file, 'r') as f:
            json_data = json.load(f)

        for record_name, case_dict in json_data.items():
            for data_dict in case_dict['images']:
                # image_data = {
                #     'image_path': image_path,
                #     'drone_path': os.path.join(drone_source_data_folder, image_name),
                #     'annotations': []
                # }
                # json_data[image_name] = image_data
                original_source_file_relpath, original_source_index = source_identifier_to_path_and_index(data_dict['original_source_identifier'])

                if original_source_file_relpath.startswith(prefix:='deschutes/'):
                    original_source_file_relpath = 'teams/deschutes/' + original_source_file_relpath[len(prefix):]
                if ';' in original_source_file_relpath:
                    subpaths = original_source_file_relpath.split(';')
                    original_source_file_relpath = subpaths[original_source_index]
                    original_source_index = 0
                if original_source_file_relpath.startswith(ap:='../projects/eagle_eyes_video_scanner/assets'):
                    original_source_file_relpath = original_source_file_relpath.replace(ap, '__ASSET_PATH__')
                else:
                    original_source_file_relpath = os.path.join('___DRONE_PATH___', original_source_file_relpath)

                original_source_file = original_source_file_relpath.replace('___DRONE_PATH___', drone_source_data_folder).replace('__ASSET_PATH__', asset_path_for_testing(''))

                assert os.path.exists(original_source_file), f"File {original_source_file} does not exist"

                print(f"Getting source file: {original_source_file}: {original_source_index}")
                fsii = FrameSourceInfoAndImage.from_frame_source_info(FrameSourceInfo(
                        source_file=original_source_file,
                        source_file_id=create_ulid_for_file(original_source_file),
                        source_index=original_source_index,
                        annotations=[Annotation(**annotation) for annotation in data_dict['annotations']],
                        geodata=get_source_geodata_or_none(source_file=original_source_file, source_index=original_source_index),
                        record_query=RecordQuery(
                            case=record_name,
                            record_id=None,
                            nickname=record_name,
                            file=original_source_file
                        )
                    )
                )
                fsii.save_to_image_file(temp_new_dataset_folder)



if __name__ == '__main__':
    save_dataset_in_new_format()
