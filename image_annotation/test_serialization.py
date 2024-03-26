from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple, List

import msgpack_numpy
import numpy as np
import pytest

from image_annotation.serialization import DataClassWithNumpyPreSerializer, MsgPackSerializer, FileSerializer, FileWithMetaDataSer, \
    ISerializeableWithTransients
from artemis.general.custom_types import BGRImageArray
from artemis.general.utils_for_testing import hold_tempfile
from artemis.image_processing.image_utils import create_random_image
from dataclasses_serialization.json import JSONSerializer


@dataclass
class ChildClass:
    age: int = 0


@dataclass
class ParentClass:
    children: Dict[str, ChildClass]


def test_dataclass_serialization():
    parent = ParentClass(children={
        'bobby': ChildClass(4),
        'suzy': ChildClass(5),
    })
    serobj = JSONSerializer.serialize(parent)

    obj = JSONSerializer.deserialize(ParentClass, serobj)

    assert obj.children['suzy'].age == 5


@dataclass
class People:
    name: str
    img: BGRImageArray


@dataclass
class TheBookOfFaces:
    database: Dict[int, People]


def test_serialization_of_numpy_object():
    obj = TheBookOfFaces({
        364226: People(name='Nancy', img=create_random_image((30, 40))),
        32532234: People(name='Abdul', img=create_random_image((30, 40))),
    })

    # Object -> primitive -> object
    primitive_object = DataClassWithNumpyPreSerializer.serialize(obj)
    assert isinstance(primitive_object, dict)
    deser = DataClassWithNumpyPreSerializer.deserialize(TheBookOfFaces, primitive_object)
    assert np.array_equal(obj.database[364226].img, deser.database[364226].img)

    # Object -> primitiver -> bytes -> primitive -> object
    ser = msgpack_numpy.dumps(primitive_object)
    assert isinstance(ser, bytes)
    reloaded_primitive_object = msgpack_numpy.loads(ser, strict_map_key=False)
    reloaded_obj = DataClassWithNumpyPreSerializer.deserialize(TheBookOfFaces, reloaded_primitive_object)
    assert np.array_equal(obj.database[364226].img, reloaded_obj.database[364226].img)

    # Object -> bytes -> object
    msgpack_ser = MsgPackSerializer(TheBookOfFaces)
    ser = msgpack_ser.dumps(obj)
    assert isinstance(ser, bytes)
    re_re_loaded_obj = msgpack_ser.loads(ser)
    assert np.array_equal(obj.database[364226].img, re_re_loaded_obj.database[364226].img)

    # Object -> file -> object
    with hold_tempfile('.mpk') as f:
        serializer = FileSerializer.from_msgpack(path=f, klass=TheBookOfFaces)
        serializer.dump(obj)
        re_re_re_loaded_obj = serializer.load()
    assert np.array_equal(obj.database[364226].img, re_re_re_loaded_obj.database[364226].img)


def test_tuple_serialization():
    obj = (3, 'aaa', {'b': 4.0})
    serializer = MsgPackSerializer(Tuple[int, str, Dict[str, float]])
    ser = serializer.dumps(obj)
    recon_obj = serializer.loads(ser)
    assert obj == recon_obj


class SomeEnum(Enum):
    ALICE = "Alice"
    BOB = "Bob"
    Carl = "Carl"


def test_enum_serialization():
    serializer = MsgPackSerializer(List[SomeEnum])
    ser = serializer.dumps([SomeEnum.Carl, SomeEnum.ALICE])
    deser = serializer.loads(ser)
    assert deser == [SomeEnum.Carl, SomeEnum.ALICE]


@dataclass
class MyMetadata:
    n_frames: int
    size_xy: Tuple[int, int]


@dataclass
class MyHugeVideo:
    frames: List[BGRImageArray]

    def get_metadata(self) -> MyMetadata:
        return MyMetadata(n_frames=len(self.frames), size_xy=(self.frames[0].shape[1], self.frames[0].shape[0]))

    def __eq__(self, other):
        if isinstance(other, MyHugeVideo):
            if len(other.frames) == len(self.frames):
                return all(np.array_equal(im1, im2) for im1, im2 in zip(self.frames, other.frames))
        return False


def test_metadata_serialization():
    video = MyHugeVideo([create_random_image(size_xy=(400, 300)) for _ in range(100)])

    serializer = DataClassWithNumpyPreSerializer

    metaser = serializer.serialize(video.get_metadata())
    videoser = serializer.serialize(video)

    with hold_tempfile(ext='.mpk') as fname:
        with open(fname, 'wb') as f:
            packer = msgpack_numpy.Packer(f)
            f.write(packer.pack(metaser))
            f.write(packer.pack(videoser))

        with open(fname, 'rb') as f:
            unpacker = msgpack_numpy.Unpacker(f, raw=False)
            remetaser = unpacker.unpack()
            revideoser = unpacker.unpack()

    remeta = serializer.deserialize(MyMetadata, remetaser)
    revid = serializer.deserialize(MyHugeVideo, revideoser)
    assert remeta == video.get_metadata()
    assert revid == video


def test_metadata_serialization_compact():
    video = MyHugeVideo([create_random_image(size_xy=(400, 300)) for _ in range(100)])
    metadata = video.get_metadata()

    ser = FileWithMetaDataSer[MyMetadata, MyHugeVideo]()
    with hold_tempfile(ext='.mpk') as fpath:
        ser.dump(metadata=metadata, data=video, path=fpath)

        re_metadata = ser.load_metadata(cls=MyMetadata, path=fpath)
        re_data = ser.load_data(cls=MyHugeVideo, path=fpath)

    assert re_metadata == video.get_metadata()
    assert re_data == video


def test_serialization_with_custom_class():
    """ In this example we have a 'cache' field that we don't want to serialize. (And in fact causes an error if we do) """

    @dataclass
    class MyCustomClass:
        score_id_pairs: List[Tuple[float, int]]
        _cached_score_range_to_id: Dict[Tuple[float, float], List[int]] = field(default_factory=dict)

        def get_ids_for_score_range(self, score_range: Tuple[float, float]) -> List[int]:
            if score_range not in self._cached_score_range_to_id:
                self._cached_score_range_to_id[score_range] = [i for s, i in self.score_id_pairs if score_range[0] <= s < score_range[1]]
            return self._cached_score_range_to_id[score_range]

    myobj = MyCustomClass([(0.5, 1), (0.6, 2), (0.7, 3), (0.8, 4), (0.9, 5)])
    assert myobj.get_ids_for_score_range((0.6, 0.81)) == [2, 3, 4]

    # The problem
    with pytest.raises(TypeError):
        DataClassWithNumpyPreSerializer.serialize(myobj)

    # The solution
    custom_ser = DataClassWithNumpyPreSerializer.add_custom_handling(
        serializers={MyCustomClass: lambda x: {k: v for k, v in x.__dict__.items() if k != '_cached_score_range_to_id'}},
        deserializers={}
    )
    ser = custom_ser.serialize(myobj)
    assert ser == {'score_id_pairs': [(0.5, 1), (0.6, 2), (0.7, 3), (0.8, 4), (0.9, 5)]}
    deser = custom_ser.deserialize(MyCustomClass, ser)
    assert deser.get_ids_for_score_range((0.6, 0.81)) == [2, 3, 4]


def test_transient_field_drop():
    """ Another option for dealing with transient fields. """

    @dataclass
    class MyCustomClass(ISerializeableWithTransients):
        score_id_pairs: List[Tuple[float, int]]
        _cached_score_range_to_id: Dict[Tuple[float, float], List[int]] = field(default_factory=dict)

        @classmethod
        def get_transient_fields(cls: type) -> Tuple[str, ...]:
            return ['_cached_score_range_to_id']

        def get_ids_for_score_range(self, score_range: Tuple[float, float]) -> List[int]:
            if score_range not in self._cached_score_range_to_id:
                self._cached_score_range_to_id[score_range] = [i for s, i in self.score_id_pairs if score_range[0] <= s < score_range[1]]
            return self._cached_score_range_to_id[score_range]

    myobj = MyCustomClass([(0.5, 1), (0.6, 2), (0.7, 3), (0.8, 4), (0.9, 5)])
    assert myobj.get_ids_for_score_range((0.6, 0.81)) == [2, 3, 4]

    ser = DataClassWithNumpyPreSerializer.serialize(myobj)
    assert ser == {'score_id_pairs': [[0.5, 1], [0.6, 2], [0.7, 3], [0.8, 4], [0.9, 5]]}
    deser = DataClassWithNumpyPreSerializer.deserialize(MyCustomClass, ser)
    assert deser.get_ids_for_score_range((0.6, 0.81)) == [2, 3, 4]


if __name__ == "__main__":
    # test_dataclass_serialization()
    # test_serialization_of_numpy_object()
    # test_tuple_serialization()
    # test_enum_serialization()
    # test_metadata_serialization()
    # test_metadata_serialization_compact()
    # test_serialization_with_custom_class()
    test_transient_field_drop()