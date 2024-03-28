import base64
import os
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple, Any, Sequence, Union

import cv2
from more_itertools import first
from tinydb import TinyDB, Query
from tinydb.table import Document

from artemis.general.custom_types import BGRImageArray
from artemis.general.hashing import compute_fixed_hash, HashRep
from artemis.general.item_cache import CacheDict
from artemis.general.utils_for_testing import hold_tempdir
from artemis.image_processing.decorders import DecordDecorder
from artemis.image_processing.image_builder import ImageBuilder
from artemis.image_processing.image_utils import BoundingBox, BGRColors
from artemis.image_processing.video_frame import FrameGeoData
from artemis.image_processing.video_reader import VideoReader
from image_annotation.annotated_image_serialization import load_tiff_with_metadata, save_tiff_with_metadata, TiffImageMetadata, GPSInfo, copy_image_and_patch_on_metadata
from image_annotation.goto_queries import GotoQuery, RecordQuery, source_identifier_to_path_and_index
from image_annotation.basic_utils import is_image_path
from image_annotation.file_utils import get_hash_for_file
from image_annotation.serialization import DataClassWithNumpyPreSerializer


# Assuming JSONSerializer is imported from your pip package


def bytes_to_base32_string(b: bytes) -> str:
    # Encode the bytes using Base32 and return the string representation
    return base64.b32encode(b).decode('utf-8').rstrip('=')


def display_dataclass_instance_without_defaults(instance: Any) -> str:
    """ Display a dataclass instance without showing default values """
    return f'{instance.__class__.__name__}({", ".join([f"{k}={v}" for k, v in instance.__dict__.items() if v != instance.__class__.__dataclass_fields__[k].default])})'


@dataclass
class Annotation:
    """ Annotation - saved to database """
    ijhw_box: Tuple[int, int, int, int]  # (i, j, h, w) box
    label: str = ''  # Optionally a label
    value: int = 1  # true-positive, true-negative, neutral
    description: str = ''  # A description of the annotation
    tags: Tuple[str, ...] = ()  # Optional identifiers

    # source_data_id: int = None  # ForeignKey to FrameSourceInfo

    def get_hash_identifier(self) -> str:
        return compute_fixed_hash((self.ijhw_box, self.value), try_objects=True, hashrep=HashRep.BASE_32)

    def __repr__(self):
        return display_dataclass_instance_without_defaults(self)


@dataclass
class FrameSourceInfo:
    """ FrameSourceInfo - saved to database """
    source_file: str
    source_index: int = 0
    source_time: Optional[float] = None  # The time in seconds from the start of the source file
    source: str = ''  # Optionally, a source-identifier (e.g. 'visual', 'thermal')
    source_file_id: Optional[str] = None  # Optionally, a unique identifier for the source file - useful when the file has moved
    # record_name: str = ''  # Optionally, a name to identify which video or image collection this originalted from
    # record_id: str = ''  # Optionally, a unique identifier for the record
    # case_name: str = ''  # Optionally, a name to identify which case this originated from
    description: str = ''  # Optionally, a description of the frame
    annotations: Optional[List[Annotation]] = field(default=None)
    geodata: Optional[FrameGeoData] = None
    record_query: Optional[RecordQuery] = None  # Used to query records associated with this detection

    def frame_hash_identifier(self) -> str:
        return compute_fixed_hash(self.get_source_identifier(), hashrep=HashRep.BASE_32)

    def get_annotation_hash_identifiers(self, annotation_ix: int) -> str:
        ann = self.annotations[annotation_ix]
        return compute_fixed_hash((self.frame_hash_identifier, ann.get_hash_identifier()), try_objects=True, hashrep=HashRep.BASE_32)

    @classmethod
    def from_source_identifier(cls, source_identifier: str) -> 'FrameSourceInfo':
        source_file, source_index = source_identifier_to_path_and_index(source_identifier)
        return cls(source_file=source_file, source_index=int(source_index))

    @classmethod
    def from_multipath(cls, possible_multipath: str, frame_index: int, **kwargs) -> 'FrameSourceInfo':
        if ';' in possible_multipath:
            path = list(possible_multipath.split(';'))[frame_index]
            return cls(source_file=path, source_index=0, **kwargs)
            # return list(self.source_file.split(';'))[self.source_index]
        else:
            return cls(source_file=possible_multipath, source_index=frame_index, **kwargs)

    def to_query(self, annotation_ix: Optional[int] = None) -> GotoQuery:
        return GotoQuery(
            file=self.source_file,
            index=self.source_index or 0,
            record_id=self.record_query.record_id if self.record_query else None,
            source=self.source,
            case=self.record_query.case if self.record_query else None,
            detector_name=None,
            ij=None if annotation_ix is None or self.annotations is None else self.annotations[annotation_ix].ijhw_box[:2],
            zoom=None if annotation_ix is None else 2.0,
        )

    def get_source_identifier(self) -> str:
        return f'{self.source_file}:{self.source_index}'

    def load_image(self) -> BGRImageArray:
        if is_image_path(self.source_file):
            assert self.source_index == 0, "If source is an image, index must be zero."
            image = cv2.imread(self.source_file)
            assert image is not None, f"Could not load image from {self.source_file}"
            return image
        else:
            try:
                return VideoReader(self.source_file).request_frame(self.source_index).image
            except Exception as e:
                print(f"Got error when trying to read frame {self.source_index} from {self.source_file}: {e}")
                print("Lets pull out the big guns...")
                return DecordDecorder(self.source_file)[self.source_index]

@dataclass
class FrameSourceInfoAndImage:
    frame_source_info: FrameSourceInfo
    image: BGRImageArray
    _fsii_version = 1

    @classmethod
    def from_frame_source_info(cls, frame_source_info: FrameSourceInfo) -> 'FrameSourceInfoAndImage':
        return cls(frame_source_info=frame_source_info, image=frame_source_info.load_image())

    @classmethod
    def load_from_image_file(cls, path: str) -> 'FrameSourceInfoAndImage':
        image, metadata = load_tiff_with_metadata(path)
        jsonable_metadata = dict(metadata.jsonable_metadata)
        del jsonable_metadata['_fsii_version']  # If we change formats - we can use the version to help figure out how to read.
        frame_source_info = DataClassWithNumpyPreSerializer.deserialize(cls=FrameSourceInfo, serialized_obj=metadata.jsonable_metadata)
        return cls(frame_source_info=frame_source_info, image=image)

    def save_to_image_file(self, parent_dir: str) -> str:
        """
        Save to an image file.  The frame source info metadata will be
        :param parent_dir: Where to save this file to
            e.g. '/path/to/image/folder/'
        :return: The path to the saved image file, with the apprpriate extension
            e.g. '/path/to/image/folder/
        """
        base_path, ext = os.path.splitext(parent_dir)
        assert not ext, f"Please specify an extensionless path.  You specified: {parent_dir}"
        os.makedirs(parent_dir, exist_ok=True)

        # We construct the filename using content-based hashing - so that
        # a) We can share and pool files git cowithout working about name collisions
        # b) We have some way to look up the original source file if needed, if it has moved or been renamed.
        # Note... if source file is a directory (as in livestreams) or unavailable - we hash on the path
        # TODO: A more systematic way to do this - this all feels kind of ad-hoc
        source_reference_hash = bytes_to_base32_string(get_hash_for_file(self.frame_source_info.source_file)) \
            if os.path.isfile(self.frame_source_info.source_file) else \
            compute_fixed_hash(self.frame_source_info.get_source_identifier())
        remainder_hash = compute_fixed_hash(self.frame_source_info, try_objects=True)
        source_file_name, _ = os.path.splitext(os.path.basename(self.frame_source_info.source_file))
        filename = f'{source_reference_hash[:8]}{remainder_hash[:8]}_{source_file_name}'
        extensionless_path = os.path.join(parent_dir, filename)
        if not is_image_path(self.frame_source_info.source_file):
            extensionless_path += f"_{self.frame_source_info.source_index}"

        metadata_dict = DataClassWithNumpyPreSerializer.serialize(self.frame_source_info)
        metadata_dict['_fsii_version'] = self._fsii_version
        geodata = self.frame_source_info.geodata
        if geodata is not None:
            lat, long = geodata.lat_long
            dt = geodata.get_datetime(localize=True)
            metadata = TiffImageMetadata(
                date_time=dt,
                gps_info=GPSInfo(latitude=lat, longitude=long, altitude=self.frame_source_info.geodata.altitude_from_sea) if self.frame_source_info.geodata else None,
                jsonable_metadata=metadata_dict
            )
        else:
            metadata = TiffImageMetadata(date_time=None, jsonable_metadata=metadata_dict)

        # If we have not specified a particu`lar
        _, source_file_ext = os.path.split(self.frame_source_info.source_file.lower())

        we_can_just_copy_the_file = is_image_path(self.frame_source_info.source_file) and os.path.exists(self.frame_source_info.source_file)

        if we_can_just_copy_the_file:
            _, ext = os.path.splitext(self.frame_source_info.source_file)
            image_path = extensionless_path + '.ann' + ext.lower()
            copy_image_and_patch_on_metadata(source_path=self.frame_source_info.source_file, dest_path=image_path, metadata=metadata)
        else:
            image_path = extensionless_path + '.ann.tiff'
            save_tiff_with_metadata(self.image, metadata=metadata, path=image_path)
        return image_path

    def replace_annotations(self, annotations: Optional[Sequence[Annotation]]) -> 'FrameSourceInfoAndImage':
        return replace(self, frame_source_info=replace(self.frame_source_info, annotations=list(annotations)))


@dataclass
class AnnotatedImage:
    image: BGRImageArray
    annotations: List[Annotation]

    def render(self, thickness=3) -> BGRImageArray:
        builder = ImageBuilder.from_image(self.image)
        # Show existing boxes
        for annotation in (self.annotations or ()):
            i, j = annotation.ijhw_box[:2]
            builder.draw_box(BoundingBox.from_ijhw(*annotation.ijhw_box, label=annotation.label),
                             colour=self.image[i, j],
                             secondary_colour=BGRColors.BLACK,
                             show_score_in_label=False,
                             thickness=thickness,
                             )
        return builder.get_image()


# @dataclass
# class AnnotationQuery:
#     source_identifier: Optional[str] = None
#     collection_name: Optional[str] = None
#     label: Optional[str] = None


# JSONTinyDBSerializer = Serializer[Any, JSONStructure](
#     serialization_functions={
#         dict: lambda dct: dict_serialization(dct, key_serialization_func=JSONSerializer.serialize, value_serialization_func=JSONSerializer.serialize),
#         (list, tuple): lambda lst: list(map(JSONSerializer.serialize, lst)),
#         (str, int, float, bool, type(None)): noop_serialization,
#     },
#     deserialization_functions={
#         dict: lambda cls, dct: dict_deserialization(cls, dct, key_deserialization_func=JSONSerializer.deserialize, value_deserialization_func=JSONSerializer.deserialize),
#         list: lambda cls, lst: list_deserialization(cls, lst, deserialization_func=JSONSerializer.deserialize),
#         tuple: lambda cls, lst: tuple_deserialization(cls, lst, deserialization_func=JSONSerializer.deserialize),
#         (str, int, float, bool, type(None)): noop_deserialization,
#         Document: lambda cls, doc: JSONSerializer.deserialize(cls, doc)
#     }
# )


def get_fixed_hash_from_frame_source_info(source_identifier: str) -> int:
    # Use a seeded hash to ensure that the same source file always has the same hash
    return compute_fixed_hash(source_identifier, hashrep=HashRep.INT)


class AnnotationDatabaseAccessor:
    def __init__(self,
                 annotation_folder_path: str,
                 source_data_base_path: Optional[str] = None,  # When source paths are relative, this is the base path
                 thumbnail_size: Tuple[int, int] = (128, 128),
                 query_cache_size=3,  # Cache the last 3 queries
                 ):
        os.makedirs(annotation_folder_path, exist_ok=True)
        self.db = TinyDB(os.path.join(annotation_folder_path, 'db_cache.json'))
        self._thumbnail_size = thumbnail_size
        self._source_data_base_path = source_data_base_path
        self._image_folder_path = os.path.join(annotation_folder_path, 'images')
        self._thumbnail_folder_path = os.path.join(annotation_folder_path, 'thumbnails')
        os.makedirs(self._image_folder_path, exist_ok=True)
        os.makedirs(self._thumbnail_folder_path, exist_ok=True)
        self._query_cache = CacheDict(buffer_length=query_cache_size)
        self._cache_dirty = True



    def _get_image_path_from_source_identifier(self, frame_source_info: FrameSourceInfo) -> str:
        source_id_str = compute_fixed_hash(frame_source_info.get_source_identifier(), hashrep=HashRep.BASE_32)
        return os.path.join(self._image_folder_path, f'{source_id_str}.png')

    def _get_thumbnail_path_from_source_identifier(self, frame_source_info: FrameSourceInfo, annotation: Annotation) -> str:
        source_id_str = compute_fixed_hash(frame_source_info.get_source_identifier(), hashrep=HashRep.BASE_32)
        annotation_id_str = compute_fixed_hash(annotation, try_objects=True, hashrep=HashRep.BASE_32)
        return os.path.join(self._thumbnail_folder_path, f'{source_id_str}_{annotation_id_str}.png')

    def _relativize_path(self, path: str) -> str:
        return os.path.relpath(path, self._source_data_base_path) if self._source_data_base_path and path.startswith(self._source_data_base_path) else path

    def _source_identifier_to_query(self, source_identifier: str) -> Query:
        # hash_code = get_fixed_hash_from_frame_source_info(source_identifier)
        # query = Query().doc_id == hash_code
        # return query

        fs_info = FrameSourceInfo.from_source_identifier(source_identifier)
        return (Query().source_file == self._relativize_path(fs_info.source_file)) & (Query().source_index == fs_info.source_index)

    def get_n_frames_in_database(self) -> int:
        return len(self.db)

    def _insert_filename_and_fsi_into_database(self, filename: str, frame_source_info: FrameSourceInfo) -> None:
        database_key = get_fixed_hash_from_frame_source_info(frame_source_info.get_source_identifier())
        full_obj = dict(filename=filename, data=DataClassWithNumpyPreSerializer.serialize(frame_source_info))
        document = Document(full_obj, doc_id=database_key)
        self.db.upsert(document)

    def save_annotated_image(
            self,
            fsii: FrameSourceInfoAndImage,
            # frame_source_info: FrameSourceInfo,
            # image: Optional[BGRImageArray] = None,
            # save_image: bool = True,
            # image_filename: Optional[str] = None,  # Optionally, specify the filename to save the image as
            # save_thumbnails: bool = True
    ) -> str:
        """ Save the annotated image to the database - returning the full path to the saved image """
        image_path = fsii.save_to_image_file(self._image_folder_path)
        filename_key = os.path.basename(image_path)
        self._insert_filename_and_fsi_into_database(filename_key, fsii.frame_source_info)

        # # Save thumbnails
        # if save_thumbnails:
        #     assert image is not None, 'You must provide an image if you want to save thumbnails'
        #     if frame_source_info.annotations is not None:
        #         for annotation in frame_source_info.annotations:
        #             thumbnail_path = self._get_thumbnail_path_from_source_identifier(frame_source_info, annotation)
        #             i, j = annotation.ijhw_box[0:2]
        #             thumbnail = BoundingBox.from_ijhw(i, j, *self._thumbnail_size).crop_image(image)
        #             os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
        #             cv2.imwrite(thumbnail_path, thumbnail)

        print(f"Saved {fsii.frame_source_info.source_file} to database along with {len(fsii.frame_source_info.annotations)} annotations")
        self._clear_caches()

        return image_path

    #
    def _clear_caches(self):
        #     # self.db.storage.flush()
        #     print(f"Cleared caches on object id {id(self)}")
        self._query_cache.clear()

    def update_cache(self, full_clean: bool = False):
        """ Update the tinydb cache to make sure it totally matches the image folder.  This is useful if you've added or removed images outside of this accessor."""
        if full_clean:
            self._query_cache.clear()
        filenames = {f for f in os.listdir(self._image_folder_path) if is_image_path(f)}
        filenames_in_db = {doc['filename'] for doc in self.db.all()}
        # Remove any files in the database that are not in the folder
        self.db.remove(cond=Query().filename.one_of(filenames_in_db - filenames))
        # Add any files in the folder that are not in the database
        for filename in filenames - filenames_in_db:
            full_path = os.path.join(self._image_folder_path, filename)
            fsii = FrameSourceInfoAndImage.load_from_image_file(full_path)
            self._insert_filename_and_fsi_into_database(filename, fsii.frame_source_info)
        self._cache_dirty = False

    def lookup_frame_source_info_from_identifier(self, source_identifier: str) -> Optional[FrameSourceInfo]:
        if self._cache_dirty:
            self.update_cache()
        hash_code = get_fixed_hash_from_frame_source_info(source_identifier)
        full_json = self.db.get(doc_id=hash_code)
        if full_json is None:
            return None
        frame_source_info_json = full_json['data']
        return DataClassWithNumpyPreSerializer.deserialize(FrameSourceInfo, frame_source_info_json)

    def load_frame_source_info_and_image(self, frame_source_info_id: str) -> Optional[FrameSourceInfoAndImage]:
        if self._cache_dirty:
            self.update_cache()
        database_key = get_fixed_hash_from_frame_source_info(frame_source_info_id)
        full_json = self.db.get(doc_id=database_key)
        if full_json is None:
            return None
        filename_key = full_json['filename']
        image_path = os.path.join(self._image_folder_path, filename_key)
        return FrameSourceInfoAndImage.load_from_image_file(image_path)
        #
        # key = get_fixed_hash_from_frame_source_info(frame_source_info_id)
        # frame_source_info_json = self.db.get(doc_id=key)
        # if frame_source_info_json is None:
        #     return None
        # frame_source_info = DataClassWithNumpyPreSerializer \
        #     .deserialize(FrameSourceInfo, frame_source_info_json)
        # image_path = self._get_image_path_from_source_identifier(frame_source_info)
        # image = cv2.imread(image_path)
        # if image is None:
        #     return None
        # return FrameSourceInfoAndImage.load_from_image_file()

    def load_annotated_image(self, frame_source_info_id: str) -> Optional[AnnotatedImage]:
        """ Convenience method to load an annotated image from the database """
        frame_source_info_and_image = self.load_frame_source_info_and_image(frame_source_info_id)
        return AnnotatedImage(image=frame_source_info_and_image.image, annotations=frame_source_info_and_image.frame_source_info.annotations) \
            if frame_source_info_and_image else None

    def query_annotation_data(self, query: Optional[Query] = None) -> List[FrameSourceInfo]:
        """
        :param query: The tinydb query object, e.g. Query().data.source_file == 'some_file.jpg'.  Full structure
            Query()
                .data
                    .source_file == 'some_file.jpg'
                    .description == 'A description'
                    .annotations
                        .label == 'car'
                        .description == 'A car'
                        ... (See Annotation)
                    ... (See FrameSourceInfo)
                .filename == 'some_file.ann.jpg'
        :return: A list of FrameSourceInfo objects that match the query
        """
        if query is None:
            full_info = self.db.all()
            return [DataClassWithNumpyPreSerializer.deserialize(FrameSourceInfo, doc['data']) for doc in full_info]
        else:
            full_info = self.db.search(query)
            return [DataClassWithNumpyPreSerializer.deserialize(FrameSourceInfo, doc['data']) for doc in full_info]

    def query_text_in_any_field(self, text: str) -> List[FrameSourceInfo]:
        query = Query()
        text = text.strip()
        if not text:
            return self.query_annotation_data()
        else:
            # TODO: Enable searching by id-prefixes
            return self.query_annotation_data(
                query.data.record_query.nickname.search(text) |
                query.data.source_file.search(text) |
                query.data.annotations.any(query.label.search(text)) |
                query.data.annotations.any(query.description.search(text)) |
                query.data.annotations.any(query.tags.any(text))
            )

    def query_annotation_data_from_path_and_source_index(self, path: str, source_index: int) -> Optional[FrameSourceInfo]:
        fsis = self.query_annotation_data_from_path(path)
        return first((fsi for fsi in fsis if fsi.source_index == source_index), default=None)

    def query_annotation_data_from_path(self, path: str) -> Sequence[FrameSourceInfo]:
        """
        Find annotations from the given path.
        :param path: Can come in 2 forms:
            - A single path to a video file
            - A ;-separated list of paths to a list of images
        :return: A list of frame source info objects.
            In the case of multiple images, the source_index field will be modified to indicate the index of the image in the list.
        """
        # print(f'Running query on object id {id(self)}')
        if path in self._query_cache:
            return self._query_cache[path]

        # self.db.clear_cache()
        paths = path.split(';')
        shortlist_json = self.db.search(Query().data.source_file.one_of(paths))
        shortlist: Sequence[FrameSourceInfo] = [DataClassWithNumpyPreSerializer.deserialize(FrameSourceInfo, doc['data']) for doc in shortlist_json]
        if len(paths) == 1:
            result = shortlist
        else:
            path_to_index = {p: i for i, p in enumerate(paths)}
            result = [replace(fsi, source_file=path, source_index=path_to_index[fsi.source_file]) for fsi in shortlist]

        self._query_cache[path] = result
        print(f"Query found {len(result)} annotated frames in path {path}")
        return result

    def delete_annotation_data(self, source_identifier: Union[str, Query]) -> None:

        # query = self._source_identifier_to_query(source_identifier) if isinstance(source_identifier, str) else source_identifier
        # self.db.remove(query)
        hash_code = get_fixed_hash_from_frame_source_info(source_identifier)
        full_json = self.db.get(doc_id=hash_code)
        # object = DataClassWithNumpyPreSerializer.deserialize(FrameSourceInfo, full_json['data'])
        self.db.remove(doc_ids=[hash_code])

        # Delete image
        annotation_path = os.path.join(self._image_folder_path, full_json['filename'])
        os.remove(annotation_path)

        # for fsi in [object['filename']]:
        #     image_path = self._get_image_path_from_source_identifier(fsi)
        #     try:
        #         os.remove(image_path)
        #     except FileNotFoundError:
        #         pass
        #
        #     if fsi.annotations is None:  # Shouldn't happen but whatevs
        #         break
        #
        #     # Delete thumbnails
        #     thumbnail_paths = [self._get_thumbnail_path_from_source_identifier(fsi, annotation) for annotation in fsi.annotations]
        #     for thumbnail_path in thumbnail_paths:
        #         try:
        #             os.remove(thumbnail_path)
        #         except FileNotFoundError:
        #             pass
        # Delete from mongoDB
        # db.delete_one(query)

        self._clear_caches()  # Need to do this to avoid a bug in tinydb

        # fsi = FrameSourceInfo.from_source_identifier(source_identifier)

        # if query is None:
        #     self.db.purge()
        # else:
        # raise NotImplementedError('Deleting annotation data is not yet implemented')


DB_ACCESSOR_SINGLETON: Optional[AnnotationDatabaseAccessor] = None


def get_annotation_db_singleton() -> AnnotationDatabaseAccessor:
    if DB_ACCESSOR_SINGLETON is None:
        raise Exception('You must call hold_annotation_db_singleton before using the singleton')
    return DB_ACCESSOR_SINGLETON


@contextmanager
def hold_annotation_db_singleton(annotation_folder_path: str, source_data_base_path: Optional[str] = None, thumbnail_size: Tuple[int, int] = (128, 128)) -> AnnotationDatabaseAccessor:
    global DB_ACCESSOR_SINGLETON
    if DB_ACCESSOR_SINGLETON is not None:
        raise Exception('You can only hold one db accessor singleton at a time')
    DB_ACCESSOR_SINGLETON = AnnotationDatabaseAccessor(annotation_folder_path=annotation_folder_path, source_data_base_path=source_data_base_path, thumbnail_size=thumbnail_size)
    try:
        yield DB_ACCESSOR_SINGLETON
    finally:
        DB_ACCESSOR_SINGLETON = None


@contextmanager
def hold_temp_annotation_db_singleton() -> AnnotationDatabaseAccessor:
    """ For testing """
    with hold_tempdir() as annotation_folder_path:
        with hold_annotation_db_singleton(annotation_folder_path) as db_accessor:
            yield db_accessor


# Example usage
if __name__ == "__main__":
    db_accessor = AnnotationDatabaseAccessor('/path/to/source_data', '/path/to/annotations')
    # Use db_accessor for various operations
