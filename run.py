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
from sklearn.tree import DecisionTreeClassifier

mora_phoneme_list = ["a", "i", "u", "e", "o", "I", "U", "E", "N", "cl", "pau"]
voiced_phoneme_list = (
    ["a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N"]
    + ["n", "m", "y", "r", "w", "g", "z", "j", "d", "b"]
    + ["ny", "my", "ry", "gy", "by"]
)
unvoiced_mora_phoneme_list = ["A", "I", "U", "E", "O", "cl", "pau"]


def f0_mean(
    f0: numpy.ndarray,
    rate: float,
    split_second_list: List[float],
    weight: numpy.ndarray,
):
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    output = numpy.array(
        [
            numpy.sum(a[~numpy.isnan(a)] * b[~numpy.isnan(a)])
            / numpy.sum(b[~numpy.isnan(a)])
            for a, b in zip(numpy.split(f0, indexes), numpy.split(weight, indexes))
        ]
    )
    return output


def voiced_ratio(f0: numpy.ndarray, rate: float, split_second_list: List[float]):
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    return numpy.array([numpy.mean(~numpy.isnan(a)) for a in numpy.split(f0, indexes)])


def loudness_mean(loudness: numpy.ndarray, rate: float, split_second_list: List[float]):
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    return numpy.array([numpy.mean(a) for a in numpy.split(loudness, indexes)])


def stride_array(array: numpy.ndarray, sampling_length: int, padding_value: float):
    return numpy.lib.stride_tricks.as_strided(
        numpy.pad(array, sampling_length // 2, constant_values=padding_value),
        shape=(len(array) + (sampling_length + 1) % 2, sampling_length),
        strides=array.strides + array.strides,
    )


def diff_array(
    array: numpy.ndarray, stride: int, sampling_length: int, padding_value: float
):
    diff = array[:-stride] - array[stride:]
    return numpy.lib.stride_tricks.as_strided(
        numpy.pad(diff, stride, constant_values=padding_value),
        shape=(len(array), sampling_length),
        strides=array.strides + (array.strides[0] * stride,),
    )


def pad_diff_array(
    array: numpy.ndarray, stride: int, sampling_length: int, padding_value: float
):
    array2 = numpy.pad(array, stride, constant_values=padding_value)
    diff = array2[:-stride] - array2[stride:]
    return numpy.lib.stride_tricks.as_strided(
        diff,
        shape=(len(array), sampling_length),
        strides=array.strides + (array.strides[0] * stride,),
    )


@dataclass
class InputData:
    name: str
    f0: SamplingData
    phoneme_list: List[JvsPhoneme]
    loudness: SamplingData
    accent_start: List[bool]
    accent_end: List[bool]
    accent_phrase_start: List[bool]
    accent_phrase_end: List[bool]


def pre_process(datas: List[InputData], sampling_length: int):
    xs: List[numpy.ndarray] = []
    mora_accent_starts = []
    mora_accent_ends = []
    for d in datas:
        mora_indexes = [
            i for i, p in enumerate(d.phoneme_list) if p.phoneme in mora_phoneme_list
        ]

        mora_accent_starts.append([d.accent_start[i] for i in mora_indexes])
        mora_accent_ends.append([d.accent_end[i] for i in mora_indexes])

        mora_accent_phrase_start = numpy.array(
            [d.accent_phrase_start[i] for i in mora_indexes]
        )
        mora_accent_phrase_end = numpy.array(
            [d.accent_phrase_end[i] for i in mora_indexes]
        )

        rate = d.f0.rate
        f0 = d.f0.array
        f0[f0 == 0] = numpy.nan
        f0 = (f0 - numpy.nanmean(f0)) / numpy.sqrt(numpy.nanstd(f0))

        loudness = d.loudness.resample(rate)

        min_length = min(len(f0), len(loudness))
        f0 = f0[:min_length]
        loudness = loudness[:min_length]

        phoneme_f0 = f0_mean(
            f0=f0,
            rate=rate,
            split_second_list=[p.end for p in d.phoneme_list[:-1]],
            weight=loudness,
        )
        phoneme_f0[numpy.isnan(phoneme_f0)] = -10

        phoneme_length = [p.end - p.start for p in d.phoneme_list]
        mora_f0 = numpy.array([], dtype=numpy.float32)
        for i, diff in enumerate(numpy.diff(numpy.r_[0, mora_indexes])):
            index = mora_indexes[i]
            if (
                diff == 1
                or d.phoneme_list[index - 1].phoneme not in voiced_phoneme_list
            ):
                mora_f0 = numpy.r_[mora_f0, phoneme_f0[index]]
            else:
                a = phoneme_f0[index - 1] * phoneme_length[index - 1]
                b = phoneme_f0[index] * phoneme_length[index]
                mora_f0 = numpy.r_[
                    mora_f0,
                    (a + b) / (phoneme_length[index] + phoneme_length[index - 1]),
                ]

        mora_f0[
            [
                d.phoneme_list[i].phoneme in unvoiced_mora_phoneme_list
                for i in mora_indexes
            ]
        ] = -10

        mora_voiced_ratio = voiced_ratio(
            f0=f0,
            rate=rate,
            split_second_list=[d.phoneme_list[i].end for i in mora_indexes][:-1],
        )
        # mora_loudness = loudness_mean(
        #     loudness=loudness, rate=rate, split_second_list=mora_seconds[:-1]
        # )

        x = numpy.concatenate(
            [
                # stride_array(
                #     array=mora_f0,
                #     sampling_length=1,
                #     padding_value=-10,
                # ),
                # stride_array(
                #     array=mora_voiced_ratio,
                #     sampling_length=sampling_length,
                #     padding_value=0,
                # ),
                stride_array(
                    array=mora_f0[1:] - mora_f0[:-1],
                    sampling_length=sampling_length - 1,
                    padding_value=0,
                ),
                # *[
                #     diff_array(
                #         array=mora_f0,
                #         stride=i + 1,
                #         sampling_length=2,
                #         padding_value=0,
                #     )
                #     for i in range(sampling_length // 2)
                # ],
                stride_array(
                    array=mora_accent_phrase_start,
                    sampling_length=sampling_length,
                    padding_value=0,
                ),
                stride_array(
                    array=mora_accent_phrase_end,
                    sampling_length=sampling_length,
                    padding_value=0,
                ),
                # stride_array(
                #     array=mora_loudness,
                #     sampling_length=sampling_length,
                #     padding_value=0,
                # ),
                # stride_array(
                #     array=mora_loudness[1:] - mora_loudness[:-1],
                #     sampling_length=sampling_length - 1,
                #     padding_value=0,
                # ),
                # pad_diff_array(
                #     array=mora_loudness,
                #     stride=2,
                #     sampling_length=2,
                #     padding_value=0,
                # ),
            ],
            axis=1,
        )
        xs.append(x)

    X = numpy.concatenate(xs)
    start_y = numpy.array(
        list(chain.from_iterable(mora_accent_starts)), dtype=numpy.int32
    )
    end_y = numpy.array(list(chain.from_iterable(mora_accent_ends)), dtype=numpy.int32)
    y = start_y + numpy.roll(end_y, 1) * 2
    split_index = numpy.cumsum([len(a) for a in mora_accent_starts[:-1]])
    return X, y, split_index


def create_data(
    f0_dir: Path,
    phoneme_list_dir: Path,
    loudness_dir: Path,
    accent_start_dir: Path,
    accent_end_dir: Path,
    accent_phrase_start_dir: Path,
    accent_phrase_end_dir: Path,
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

    loudness_paths = sorted(loudness_dir.rglob("*.npy"))
    if data_num is not None:
        loudness_paths = loudness_paths[:data_num]
    assert len(f0_paths) == len(loudness_paths)

    accent_start_paths = sorted(accent_start_dir.rglob("*.txt"))
    if data_num is not None:
        accent_start_paths = accent_start_paths[:data_num]
    assert len(f0_paths) == len(accent_start_paths)

    accent_end_paths = sorted(accent_end_dir.rglob("*.txt"))
    if data_num is not None:
        accent_end_paths = accent_end_paths[:data_num]
    assert len(f0_paths) == len(accent_end_paths)

    accent_phrase_start_paths = sorted(accent_phrase_start_dir.rglob("*.txt"))
    if data_num is not None:
        accent_phrase_start_paths = accent_phrase_start_paths[:data_num]
    assert len(f0_paths) == len(accent_phrase_start_paths)

    accent_phrase_end_paths = sorted(accent_phrase_end_dir.rglob("*.txt"))
    if data_num is not None:
        accent_phrase_end_paths = accent_phrase_end_paths[:data_num]
    assert len(f0_paths) == len(accent_phrase_end_paths)

    datas = [
        InputData(
            name=f0_path.stem,
            f0=SamplingData.load(f0_path),
            phoneme_list=JvsPhoneme.load_julius_list(phoneme_list_path),
            loudness=SamplingData.load(loudness_path),
            accent_start=[bool(int(s)) for s in accent_start_path.read_text().split()],
            accent_end=[bool(int(s)) for s in accent_end_path.read_text().split()],
            accent_phrase_start=[
                bool(int(s)) for s in accent_phrase_start_path.read_text().split()
            ],
            accent_phrase_end=[
                bool(int(s)) for s in accent_phrase_end_path.read_text().split()
            ],
        )
        for (
            f0_path,
            phoneme_list_path,
            loudness_path,
            accent_start_path,
            accent_end_path,
            accent_phrase_start_path,
            accent_phrase_end_path,
        ) in zip(
            f0_paths,
            phoneme_list_paths,
            loudness_paths,
            accent_start_paths,
            accent_end_paths,
            accent_phrase_start_paths,
            accent_phrase_end_paths,
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
    loudness_dir: Path,
    accent_start_dir: Path,
    accent_end_dir: Path,
    accent_phrase_start_dir: Path,
    accent_phrase_end_dir: Path,
    output_path: Path,
    output_score_path: Path,
    data_num: Optional[int],
    speaker_valid_filter: Optional[str],
    utterance_valid_filter: Optional[str],
    sampling_length: int,
    dtc_min_samples_split: float,
    dtc_min_samples_leaf: float,
    seed: int,
):
    config = deepcopy(locals())

    train_datas, valid_datas = create_data(
        f0_dir=f0_dir,
        phoneme_list_dir=phoneme_list_dir,
        loudness_dir=loudness_dir,
        accent_start_dir=accent_start_dir,
        accent_end_dir=accent_end_dir,
        accent_phrase_start_dir=accent_phrase_start_dir,
        accent_phrase_end_dir=accent_phrase_end_dir,
        speaker_valid_filter=speaker_valid_filter,
        utterance_valid_filter=utterance_valid_filter,
        data_num=data_num,
    )

    train_X, train_y, train_split_index = pre_process(
        datas=train_datas, sampling_length=sampling_length
    )
    valid_X, valid_y, valid_split_index = pre_process(
        datas=valid_datas, sampling_length=sampling_length
    )
    model = DecisionTreeClassifier(
        class_weight="balanced",
        min_samples_split=dtc_min_samples_split,
        min_samples_leaf=dtc_min_samples_leaf,
        random_state=seed,
    )
    model.fit(train_X, train_y)

    # train accuracy
    random_state = numpy.random.RandomState(seed)
    idx = random_state.permutation(len(train_datas))[: len(valid_datas)]
    test_train_X, test_train_y, _ = pre_process(
        datas=[train_datas[i] for i in idx], sampling_length=sampling_length
    )
    train_predicted = model.predict(test_train_X)
    train_start_precision = precision_score(test_train_y == 1, train_predicted == 1)
    train_start_recall = recall_score(test_train_y == 1, train_predicted == 1)
    train_start_accuracy = accuracy_score(test_train_y == 1, train_predicted == 1)
    train_end_precision = precision_score(test_train_y == 2, train_predicted == 2)
    train_end_recall = recall_score(test_train_y == 2, train_predicted == 2)
    train_end_accuracy = accuracy_score(test_train_y == 2, train_predicted == 2)

    # vaild accuracy
    valid_predicted = model.predict(valid_X)
    valid_start_precision = precision_score(valid_y == 1, valid_predicted == 1)
    valid_start_recall = recall_score(valid_y == 1, valid_predicted == 1)
    valid_start_accuracy = accuracy_score(valid_y == 1, valid_predicted == 1)
    valid_end_precision = precision_score(valid_y == 2, valid_predicted == 2)
    valid_end_recall = recall_score(valid_y == 2, valid_predicted == 2)
    valid_end_accuracy = accuracy_score(valid_y == 2, valid_predicted == 2)

    valid_prob = model.predict_proba(valid_X)
    obj = {
        data.name: predicted
        for predicted, data in zip(
            numpy.split(valid_prob, valid_split_index), valid_datas
        )
    }
    numpy.save(output_path, obj)

    df = DataFrame(
        [
            dict(
                train_start_precision=train_start_precision,
                train_start_recall=train_start_recall,
                train_start_accuracy=train_start_accuracy,
                train_end_precision=train_end_precision,
                train_end_recall=train_end_recall,
                train_end_accuracy=train_end_accuracy,
                valid_start_precision=valid_start_precision,
                valid_start_recall=valid_start_recall,
                valid_start_accuracy=valid_start_accuracy,
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
    parser.add_argument("--loudness_dir", type=Path, required=True)
    parser.add_argument("--accent_start_dir", type=Path, required=True)
    parser.add_argument("--accent_end_dir", type=Path, required=True)
    parser.add_argument("--accent_phrase_start_dir", type=Path, required=True)
    parser.add_argument("--accent_phrase_end_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--output_score_path", type=Path, required=True)
    parser.add_argument("--data_num", type=int)
    parser.add_argument("--speaker_valid_filter", type=str)
    parser.add_argument("--utterance_valid_filter", type=str)
    parser.add_argument("--sampling_length", type=int, required=True)
    parser.add_argument("--dtc_min_samples_split", type=float, required=True)
    parser.add_argument("--dtc_min_samples_leaf", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    run(**vars(parser.parse_args()))
