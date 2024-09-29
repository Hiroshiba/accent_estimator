import numpy

vowels = ["pau", "a", "i", "u", "e", "o", "n", "cl"]


def vowel_to_id(vowel: str):
    if vowel == "sil":
        vowel = "pau"
    vowel = vowel.lower()
    return vowels.index(vowel)


def generate_position_array(length: int):
    return numpy.concatenate(
        [
            numpy.linspace(-1, 1, length).reshape(length, 1),
            numpy.linspace(1, -1, length).reshape(length, 1),
        ],
        axis=1,
    )
