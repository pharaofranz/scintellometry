# Helper functions for VLBI readers (VDIF, Mark5B).
import struct
import warnings

from astropy.utils import OrderedDict, lazyproperty

OPTIMAL_2BIT_HIGH = 3.3359
eight_word_struct = struct.Struct('<8I')
four_word_struct = struct.Struct('<4I')


def make_parser(word_index, bit_index, bit_length):
    """Convert specific bits from a header word to a bool or integer."""
    if bit_length == 1:
        return lambda words: bool((words[word_index] >> bit_index) & 1)
    elif bit_length == 32:
        assert bit_index == 0
        return lambda words: words[word_index]
    else:
        mask = (1 << bit_length) - 1  # e.g., bit_length=8 -> 0xff
        if bit_index == 0:
            return lambda words: words[word_index] & mask
        else:
            return lambda words: (words[word_index] >> bit_index) & mask


def make_setter(word_index, bit_index, bit_length, default=None):
    def setter(words, value):
        if value is None and default is not None:
            value = default
        value = int(value)
        word = words[word_index]
        bit_mask = (1 << bit_length) - 1
        # Check that value will fit within the bit limits.
        if value & bit_mask != value:
            raise ValueError("{0} cannot be represented with {1} bits"
                             .format(value, bit_length))
        # Zero the part to be set.
        bit_mask <<= bit_index
        word = (word | bit_mask) ^ bit_mask
        # Add the value
        word |= value << bit_index
        return words[:word_index] + (word,) + words[word_index+1:]
    return setter


class HeaderPropertyGetter(object):
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner_cls):
        return HeaderProperty(instance, self.getter)


class HeaderProperty(object):
    """Mimic a dictionary, calculating entries from header words."""
    def __init__(self, header_parser, getter):
        self.header_parser = header_parser
        self.getter = getter

    def __getitem__(self, item):
        definition = self.header_parser[item]
        return self.getter(definition)

    def __getattr__(self, attr):
        try:
            return super(HeaderProperty, self).__getattr__(attr)
        except AttributeError:
            return getattr(self.header_parser)


class HeaderParser(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(HeaderParser, self).__init__(*args, **kwargs)
        # In principle, we could calculate the parsers on the fly,
        # like we do for the setters, but this would be needlessly slow,
        # so we precalculate all of them, using a dict for even better speed.
        self.parsers = {k: make_parser(*v[:3]) for k, v in self.items()}

    def __add__(self, other):
        if not isinstance(other, HeaderParser):
            return NotImplemented
        result = self.copy()
        result.update(other)
        return result

    defaults = HeaderPropertyGetter(
        lambda definition: definition[3] if len(definition) > 3 else None)

    setters = HeaderPropertyGetter(
        lambda definition: make_setter(*definition))

    def update(self, other):
        if not isinstance(other, HeaderParser):
            raise TypeError("Can only update using a HeaderParser instance.")
        super(HeaderParser, self).update(other)
        # Update the parsers rather than recalculate all the functions.
        self.parsers.update(other.parsers)


def bcd_decode(bcd):
    result = 0
    factor = 1
    while bcd > 0:
        result += (bcd & 0xf) * factor
        factor *= 10
        bcd >>= 4
    return result


def get_frame_rate(fh, header_class, thread_id=None):
    """Returns the number of frames

    Can be for a specific thread_id (by default just the first thread in
    the first header).
    """
    fh.seek(0)
    header = header_class.fromfile(fh)
    assert header['frame_nr'] == 0
    sec0 = header.seconds
    if thread_id is None and 'thread_id' in header:
        thread_id = header['thread_id']
    while header['frame_nr'] == 0:
        fh.seek(header.payloadsize, 1)
        header = header_class.fromfile(fh)
    while header['frame_nr'] > 0:
        max_frame = header['frame_nr']
        fh.seek(header.payloadsize, 1)
        header = header_class.fromfile(fh)

    if header.seconds != sec0 + 1:
        warnings.warn("Header time changed by more than 1 second?")

    return max_frame + 1
