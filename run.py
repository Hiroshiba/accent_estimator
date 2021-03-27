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
from sklearn.tree import DecisionTreeClassifier, export_graphviz

mora_phoneme_list = ["a", "i", "u", "e", "o", "I", "U", "E", "N", "cl", "pau"]


def f0_mean(f0: numpy.ndarray, rate: float, split_second_list: List[float]):
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    output = numpy.array([numpy.nanmean(a) for a in numpy.split(f0, indexes)])
    output[numpy.isnan(output)] = -10
    return output


def voiced_ratio(f0: numpy.ndarray, rate: float, split_second_list: List[float]):
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    return numpy.array([numpy.mean(~numpy.isnan(a)) for a in numpy.split(f0, indexes)])


def stride_array(array: numpy.ndarray, sampling_length: int, padding_value: float):
    return numpy.lib.stride_tricks.as_strided(
        numpy.pad(array, sampling_length // 2, constant_values=padding_value),
        shape=(len(array) + (sampling_length + 1) % 2, sampling_length),
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
                stride_array(
                    array=mora_f0_mean,
                    sampling_length=sampling_length,
                    padding_value=-10,
                ),
                stride_array(
                    array=mora_voiced_ratio,
                    sampling_length=sampling_length,
                    padding_value=0,
                ),
                stride_array(
                    array=mora_f0_mean[1:] - mora_f0_mean[:-1],
                    sampling_length=sampling_length - 1,
                    padding_value=0,
                ),
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


def train(
    output_path: Path,
    output_graph_path: Path,
    sampling_length: int,
    dtc_min_samples_split: float,
    dtc_min_samples_leaf: float,
    seed: int,
    train_X: numpy.ndarray,
    train_y: numpy.ndarray,
    valid_X: numpy.ndarray,
    valid_y: numpy.ndarray,
    valid_split_index: numpy.ndarray,
    valid_datas: List[InputData],
):
    model = DecisionTreeClassifier(
        class_weight="balanced",
        min_samples_split=dtc_min_samples_split,
        min_samples_leaf=dtc_min_samples_leaf,
        random_state=seed,
    )
    model.fit(train_X, train_y)

    # train_predicted = model.predict(train_X)
    # train_precision = precision_score(train_y, train_predicted)
    # train_recall = recall_score(train_y, train_predicted)
    # train_accuracy = accuracy_score(train_y, train_predicted)

    valid_predicted = model.predict(valid_X)
    valid_precision = precision_score(valid_y, valid_predicted)
    valid_recall = recall_score(valid_y, valid_predicted)
    valid_accuracy = accuracy_score(valid_y, valid_predicted)

    valid_prob = model.predict_proba(valid_X)
    obj = {
        data.name: predicted
        for predicted, data in zip(
            numpy.split(valid_prob, valid_split_index), valid_datas
        )
    }
    numpy.save(output_path, obj)

    export_graphviz(
        model,
        out_file=str(output_graph_path),
        feature_names=(
            [f"f0_{i-sampling_length//2}" for i in range(sampling_length)]
            + [f"vuv_{i-sampling_length//2}" for i in range(sampling_length)]
            + [
                f"f0diff_{i-sampling_length//2}_{i+1-sampling_length//2}"
                for i in range(sampling_length - 1)
            ]
        ),
        class_names=["F", "T"],
        filled=True,
        rounded=True,
    )

    return valid_precision, valid_recall, valid_accuracy


def run(
    f0_dir: Path,
    phoneme_list_dir: Path,
    accent_start_dir: Path,
    accent_end_dir: Path,
    output_start_path: Path,
    output_end_path: Path,
    output_start_graph_path: Path,
    output_end_graph_path: Path,
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

    valid_start_precision, valid_start_recall, valid_start_accuracy = train(
        output_path=output_start_path,
        output_graph_path=output_start_graph_path,
        sampling_length=sampling_length,
        dtc_min_samples_split=dtc_min_samples_split,
        dtc_min_samples_leaf=dtc_min_samples_leaf,
        seed=seed,
        train_X=train_X,
        train_y=train_start_y,
        valid_X=valid_X,
        valid_y=valid_start_y,
        valid_split_index=valid_split_index,
        valid_datas=valid_datas,
    )

    valid_end_precision, valid_end_recall, valid_end_accuracy = train(
        output_path=output_end_path,
        output_graph_path=output_end_graph_path,
        sampling_length=sampling_length,
        dtc_min_samples_split=dtc_min_samples_split,
        dtc_min_samples_leaf=dtc_min_samples_leaf,
        seed=seed,
        train_X=train_X,
        train_y=train_end_y,
        valid_X=valid_X,
        valid_y=valid_end_y,
        valid_split_index=valid_split_index,
        valid_datas=valid_datas,
    )

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
    parser.add_argument("--output_start_graph_path", type=Path, required=True)
    parser.add_argument("--output_end_graph_path", type=Path, required=True)
    parser.add_argument("--output_score_path", type=Path, required=True)
    parser.add_argument("--data_num", type=int)
    parser.add_argument("--speaker_valid_filter", type=str)
    parser.add_argument("--utterance_valid_filter", type=str)
    parser.add_argument("--sampling_length", type=int, required=True)
    parser.add_argument("--dtc_min_samples_split", type=float, required=True)
    parser.add_argument("--dtc_min_samples_leaf", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    run(**vars(parser.parse_args()))
