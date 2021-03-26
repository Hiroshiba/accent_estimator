import argparse
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import List, Optional

import numpy
from acoustic_feature_extractor.data.phoneme import JvsPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from pandas import DataFrame
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC

mora_phoneme_list = ["a", "i", "u", "e", "o", "I", "U", "E", "N", "cl", "pau"]


def f0_mean(f0: numpy.ndarray, rate: float, split_second_list: List[float]):
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    output = numpy.array([numpy.nanmean(a) for a in numpy.split(f0, indexes)])
    output[numpy.isnan(output)] = 0
    return output


def voiced_ratio(f0: numpy.ndarray, rate: float, split_second_list: List[float]):
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    return numpy.array([numpy.mean(~numpy.isnan(a)) for a in numpy.split(f0, indexes)])


def stride_array(array: numpy.ndarray, sampling_length: int):
    return numpy.lib.stride_tricks.as_strided(
        numpy.pad(array, sampling_length // 2),
        shape=(len(array), sampling_length),
        strides=array.strides + array.strides,
    )


@dataclass
class InputData:
    name: str
    f0: SamplingData
    phoneme_list: List[JvsPhoneme]
    accent_start: List[bool]
    accent_end: List[bool]


def pre_process(datas: List[InputData], sampling_length: int):
    xs: List[numpy.ndarray] = []
    mora_accent_starts = []
    mora_accent_ends = []
    for d in datas:
        mora_indexes = [
            i for i, p in enumerate(d.phoneme_list) if p.phoneme in mora_phoneme_list
        ]
        mora_seconds = [d.phoneme_list[i].end for i in mora_indexes]

        mora_accent_starts.append([d.accent_start[i] for i in mora_indexes])
        mora_accent_ends.append([d.accent_end[i] for i in mora_indexes])

        f0 = d.f0.array.copy()
        f0[f0 == 0] = numpy.nan
        norm_f0 = (f0 - numpy.nanmean(f0)) / numpy.sqrt(numpy.nanstd(f0))

        mora_f0_mean = f0_mean(
            f0=norm_f0, rate=d.f0.rate, split_second_list=mora_seconds[:-1]
        )
        mora_voiced_ratio = voiced_ratio(
            f0=norm_f0, rate=d.f0.rate, split_second_list=mora_seconds[:-1]
        )

        x = numpy.concatenate(
            [
                stride_array(array=mora_f0_mean, sampling_length=sampling_length),
                stride_array(array=mora_voiced_ratio, sampling_length=sampling_length),
            ],
            axis=1,
        )
        xs.append(x)

    X = numpy.concatenate(xs)
    start_y = numpy.array(
        list(chain.from_iterable(mora_accent_starts)), dtype=numpy.int32
    )
    end_y = numpy.array(list(chain.from_iterable(mora_accent_ends)), dtype=numpy.int32)
    split_index = numpy.cumsum([len(a) for a in mora_accent_starts[:-1]])
    return X, start_y, end_y, split_index


def create_data(
    f0_dir: Path,
    phoneme_list_dir: Path,
    accent_start_dir: Path,
    accent_end_dir: Path,
    speaker_valid_filter: Optional[str],
    utterance_valid_filter: Optional[str],
    data_num: Optional[int],
):
    f0_paths = sorted(f0_dir.rglob("*.npy"))
    if data_num is not None:
        f0_paths = f0_paths[:data_num]
    assert len(f0_paths) > 0

    phoneme_list_paths = sorted(phoneme_list_dir.rglob("*.lab"))
    if data_num is not None:
        phoneme_list_paths = phoneme_list_paths[:data_num]
    assert len(f0_paths) == len(phoneme_list_paths)

    accent_start_paths = sorted(accent_start_dir.rglob("*.txt"))
    if data_num is not None:
        accent_start_paths = accent_start_paths[:data_num]
    assert len(f0_paths) == len(accent_start_paths)

    accent_end_paths = sorted(accent_end_dir.rglob("*.txt"))
    if data_num is not None:
        accent_end_paths = accent_end_paths[:data_num]
    assert len(f0_paths) == len(accent_end_paths)

    datas = [
        InputData(
            name=f0_path.stem,
            f0=SamplingData.load(f0_path),
            phoneme_list=JvsPhoneme.load_julius_list(phoneme_list_path),
            accent_start=[bool(int(s)) for s in accent_start_path.read_text().split()],
            accent_end=[bool(int(s)) for s in accent_end_path.read_text().split()],
        )
        for f0_path, phoneme_list_path, accent_start_path, accent_end_path in zip(
            f0_paths, phoneme_list_paths, accent_start_paths, accent_end_paths
        )
    ]

    train_datas: List[InputData] = []
    valid_datas: List[InputData] = []
    for d in datas:
        if (speaker_valid_filter is not None and speaker_valid_filter in d.name) or (
            utterance_valid_filter is not None and utterance_valid_filter in d.name
        ):
            valid_datas.append(d)
        else:
            train_datas.append(d)

    return train_datas, valid_datas


def run(
    f0_dir: Path,
    phoneme_list_dir: Path,
    accent_start_dir: Path,
    accent_end_dir: Path,
    output_start_path: Path,
    output_end_path: Path,
    output_score_path: Path,
    data_num: Optional[int],
    speaker_valid_filter: Optional[str],
    utterance_valid_filter: Optional[str],
    sampling_length: int,
    svc_c: float,
    svc_seed: int,
):
    config = deepcopy(locals())

    train_datas, valid_datas = create_data(
        f0_dir=f0_dir,
        phoneme_list_dir=phoneme_list_dir,
        accent_start_dir=accent_start_dir,
        accent_end_dir=accent_end_dir,
        speaker_valid_filter=speaker_valid_filter,
        utterance_valid_filter=utterance_valid_filter,
        data_num=data_num,
    )

    train_X, train_start_y, train_end_y, train_split_index = pre_process(
        datas=train_datas, sampling_length=sampling_length
    )
    valid_X, valid_start_y, valid_end_y, valid_split_index = pre_process(
        datas=valid_datas, sampling_length=sampling_length
    )

    start_svc = SVC(
        class_weight="balanced", verbose=True, C=svc_c, random_state=svc_seed
    )
    start_svc.fit(train_X, train_start_y)

    # train_start_predicted = start_svc.predict(train_X)
    # train_start_precision = precision_score(train_start_y, train_start_predicted)
    # train_start_recall = recall_score(train_start_y, train_start_predicted)
    # train_start_accuracy = accuracy_score(train_start_y, train_start_predicted)

    valid_start_predicted = start_svc.predict(valid_X)
    valid_start_precision = precision_score(valid_start_y, valid_start_predicted)
    valid_start_recall = recall_score(valid_start_y, valid_start_predicted)
    valid_start_accuracy = accuracy_score(valid_start_y, valid_start_predicted)

    valid_start_prob = start_svc.decision_function(valid_X)
    obj = {
        data.name: predicted
        for predicted, data in zip(
            numpy.split(valid_start_prob, valid_split_index), valid_datas
        )
    }
    numpy.save(output_start_path, obj)

    end_svc = SVC(class_weight="balanced", verbose=True, C=svc_c, random_state=svc_seed)
    end_svc.fit(train_X, train_end_y)

    # train_end_predicted = end_svc.predict(train_X)
    # train_end_precision = precision_score(train_end_y, train_end_predicted)
    # train_end_recall = recall_score(train_end_y, train_end_predicted)
    # train_end_accuracy = accuracy_score(train_end_y, train_end_predicted)

    valid_end_predicted = end_svc.predict(valid_X)
    valid_end_precision = precision_score(valid_end_y, valid_end_predicted)
    valid_end_recall = recall_score(valid_end_y, valid_end_predicted)
    valid_end_accuracy = accuracy_score(valid_end_y, valid_end_predicted)

    valid_end_prob = end_svc.decision_function(valid_X)
    obj = {
        data.name: predicted
        for predicted, data in zip(
            numpy.split(valid_end_prob, valid_split_index), valid_datas
        )
    }
    numpy.save(output_end_path, obj)

    df = DataFrame(
        [
            dict(
                # train_start_precision=train_start_precision,
                # train_start_recall=train_start_recall,
                # train_start_accuracy=train_start_accuracy,
                valid_start_precision=valid_start_precision,
                valid_start_recall=valid_start_recall,
                valid_start_accuracy=valid_start_accuracy,
                # train_end_precision=train_end_precision,
                # train_end_recall=train_end_recall,
                # train_end_accuracy=train_end_accuracy,
                valid_end_precision=valid_end_precision,
                valid_end_recall=valid_end_recall,
                valid_end_accuracy=valid_end_accuracy,
                **config,
            )
        ]
    )
    df.to_csv(output_score_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0_dir", type=Path, required=True)
    parser.add_argument("--phoneme_list_dir", type=Path, required=True)
    parser.add_argument("--accent_start_dir", type=Path, required=True)
    parser.add_argument("--accent_end_dir", type=Path, required=True)
    parser.add_argument("--output_start_path", type=Path, required=True)
    parser.add_argument("--output_end_path", type=Path, required=True)
    parser.add_argument("--output_score_path", type=Path, required=True)
    parser.add_argument("--data_num", type=int)
    parser.add_argument("--speaker_valid_filter", type=str)
    parser.add_argument("--utterance_valid_filter", type=str)
    parser.add_argument("--sampling_length", type=int, required=True)
    parser.add_argument("--svc_c", type=float, required=True)
    parser.add_argument("--svc_seed", type=int, required=True)
    run(**vars(parser.parse_args()))
