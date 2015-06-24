# Helper functions for VLBI readers (VDIF, Mark5B).
import struct
import warnings
import io

import numpy as np
from astropy.utils import OrderedDict

OPTIMAL_2BIT_HIGH = 3.3359
eight_word_struct = struct.Struct('<8I')
four_word_struct = struct.Struct('<4I')
DTYPE_WORD = np.dtype('<u4')


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


class VLBIHeaderBase(object):
    """Base class for all VLBI headers.

    Defines a number of common routines.

    Generally, the actual class should define:

      _struct: HeaderParser instance corresponding to this class.
      _header_parser: HeaderParser instance corresponding to this class.

    It also should define properties (getters *and* setters):

      payloadsize: number of bytes used by payload

      framesize: total number of bytes for header + payload

      get_time, set_time, and a corresponding time property:
           time at start of payload
    """

    def __init__(self, words, verify=True):
        if words is None:
            self.words = (0,) * (self._struct.size // 4)
        else:
            self.words = words
        if verify:
            self.verify()

    def verify(self):
        """Verify that the length of the words is consistent.

        Subclasses should override this to do more thorough checks.
        """
        assert len(self.words) == (self._struct.size // 4)

    def copy(self):
        return self.__class__(self.words, verify=False)

    @property
    def size(self):
        return self._struct.size

    @classmethod
    def frombytes(cls, s, *args, **kwargs):
        """Read VLBI Header from bytes.

        Arguments are the same as for class initialisation.
        """
        return cls(cls._struct.unpack(s), *args, **kwargs)

    def tobytes(self):
        return self._struct.pack(*self.words)

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        """Read VLBI Header from file.

        Arguments are the same as for class initialisation.
        """
        size = cls._struct.size
        s = fh.read(size)
        if len(s) != size:
            raise EOFError
        return cls.frombytes(s, *args, **kwargs)

    def tofile(self, fh):
        """Write VLBI Frame header to filehandle."""
        return fh.write(self.tobytes())

    @classmethod
    def fromvalues(cls, *args, **kwargs):
        """Initialise a header from parsed values.

        Here, the parsed values must be given as keyword arguments, i.e.,
        for any header = cls(<somedata>), cls.fromvalues(**header) == header.

        However, unlike for the 'fromkeys' class method, data can also be set
        using arguments named after header methods such 'time'.

        If any arguments are needed to initialize an empty header, those
        can be passed on in ``*args``.
        """
        # Initialize an empty header.
        self = cls(None, *args, verify=False)
        # First set all keys to keyword arguments or defaults.
        for key in self.keys():
            if key in kwargs:
                self[key] = kwargs.pop(key)
            elif self._header_parser.defaults[key] is not None:
                self[key] = self._header_parser.defaults[key]

        # Next, use remaining keyword arguments to set properties.
        # Order may be important so use list:
        for key in self._properties:
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))

        if kwargs:
            warnings.warn("Some keywords unused in header initialisation: {0}"
                          .format(kwargs))
        self.verify()
        return self

    @classmethod
    def fromkeys(cls, *args, **kwargs):
        """Like fromvalues, but without any interpretation of keywords."""
        self = cls(None, *args, verify=False)
        for key in self.keys():
            self.words = self._header_parser.setters[key](
                self.words, kwargs.pop(key))

        if kwargs:
            warnings.warn("Some keywords unused in header initialisation: {0}"
                          .format(kwargs))
        self.verify()
        return self

    def __getitem__(self, item):
        try:
            return self._header_parser.parsers[item](self.words)
        except KeyError:
            raise KeyError("{0} header does not contain {1}"
                           .format(self.__class__.__name__, item))

    def __setitem__(self, item, value):
        try:
            self.words = self._header_parser.setters[item](self.words, value)
        except KeyError:
            raise KeyError("{0} header does not contain {1}"
                           .format(self.__class__.__name__, item))

    def keys(self):
        return self._header_parser.keys()

    def __eq__(self, other):
        return (type(self) is type(other) and
                list(self.words) == list(other.words))

    def __contains__(self, key):
        return key in self.keys()

    def __repr__(self):
        name = self.__class__.__name__
        return ("<{0} {1}>".format(name, (",\n  " + len(name) * " ").join(
            ["{0}: {1}".format(k, self[k]) for k in self.keys()])))


def bcd_decode(value):
    bcd = value
    result = 0
    factor = 1
    while bcd > 0:
        digit = bcd & 0xf
        if not (0 <= digit <= 9):
            raise ValueError("Invalid BCD encoded value {0}={1}."
                             .format(value, hex(value)))
        result += digit * factor
        factor *= 10
        bcd >>= 4
    return result


def bcd_encode(value):
    result = 0
    factor = 1
    while value > 0:
        value, digit = divmod(value, 10)
        result += digit*factor
        factor *= 16
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


class VLBIPayloadBase(object):

    _size = None
    _encoders = {}
    _decoders = {}

    def __init__(self, words, nchan=1, bps=2, complex_data=False):
        """Container for decoding and encoding VDIF payloads.

        Parameters
        ----------
        words : ndarray
            Array containg LSB unsigned words (with the right size) that
            encode the payload.
        nchan : int
            Number of channels in the data.  Default: 1.
        bps : int
            Number of bits per complete sample.  Default: 2.
        complex_data : bool
            Whether data is complex or float.  Default: False.
        """
        self.words = words
        self.nchan = nchan
        self.bps = bps
        self.complex_data = complex_data
        self.nsample = len(words) * (32 // self.bps) // self.nchan
        if self._size is not None and self._size != self.size:
            raise ValueError("Encoded data should have length {0}"
                             .format(self._size))

    @classmethod
    def frombytes(cls, raw, *args, **kwargs):
        """Set paiload by interpreting bytes."""
        return cls(np.fromstring(raw, dtype=DTYPE_WORD), *args, **kwargs)

    def tobytes(self):
        """Convert payload to bytes."""
        return self.words.tostring()

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        """Read payload from file handle and decode it into data.

        Parameters
        ----------
        fh : filehandle
            Handle to the file from which data is read
        payloadsize : int
            Number of bytes to read (default: as given in ``cls._payloadsize``.

        Any other (keyword) arguments are passed on to the class initialiser.
        """
        payloadsize = kwargs.pop('payloadsize', cls._size)
        if payloadsize is None:
            raise ValueError("Payloadsize should be given as an argument "
                             "if no default is defined on the class.")
        s = fh.read(payloadsize)
        if len(s) < payloadsize:
            raise EOFError("Could not read full payload.")
        return cls.frombytes(s, *args, **kwargs)

    def tofile(self, fh):
        return fh.write(self.tobytes())

    @classmethod
    def fromdata(cls, data, bps=2):
        """Encode data as payload, using a given bits per second.

        It is assumed that the last dimension is the number of channels.
        """
        complex_data = data.dtype.kind == 'c'
        encoder = cls._encoders[bps, complex_data]
        words = encoder(data.ravel())
        return cls(words, nchan=data.shape[-1], bps=bps,
                   complex_data=complex_data)

    def todata(self, data=None):
        """Decode the payload.

        Parameters
        ----------
        data : ndarray or None
            If given, used to decode the payload into.  It should have the
            right size to store it.  Its shape is not changed.
        """
        decoder = self._decoders[self.bps, self.complex_data]
        out = decoder(self.words, out=data)
        return out.reshape(self.shape) if data is None else data

    data = property(todata, doc="Decode the payload.")

    @property
    def shape(self):
        return (self.nsample, self.nchan)

    @property
    def dtype(self):
        return np.dtype(np.complex64 if self.complex_data else np.float32)

    @property
    def size(self):
        """Size in bytes of payload."""
        return len(self.words) * DTYPE_WORD.itemsize

    def __eq__(self, other):
        return (type(self) is type(other) and
                np.all(self.words == other.words))


class VLBIFrameBase(object):

    _header_class = None
    _payload_class = None

    def __init__(self, header, payload, verify=True):
        self.header = header
        self.payload = payload
        if verify:
            self.verify()

    def verify(self):
        """Simple verification.  To be added to by subclasses."""
        assert isinstance(self.header, self._header_class)
        assert isinstance(self.payload, self._payload_class)
        assert self.payloadsize // 4 == self.payload.words.size

    @classmethod
    def frombytes(cls, raw, *args, **kwargs):
        """Read a frame set from a byte string.

        Implemented via ``fromfile`` using BytesIO.  For reading from files,
        use ``fromfile`` directly.
        """
        return cls.fromfile(io.BytesIO(raw), *args, **kwargs)

    def tobytes(self):
        return self.header.tobytes() + self.payload.tobytes()

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        verify = kwargs.pop('verify', True)
        header = cls._header_class.fromfile(fh, verify=verify)
        payload = cls._payload_class.fromfile(fh, *args, **kwargs)
        return cls(header, payload, verify)

    def tofile(self, fh):
        return fh.write(self.tobytes())

    @classmethod
    def fromdata(cls, data, header, *args, **kwargs):
        """Construct frame from data and header.

        Parameters
        ----------
        data : ndarray
            Array holding data to be encoded.
        header : VLBIHeaderBase
            Header for the frame.

        *args, **kwargs : arguments
            Additional arguments to help create the payload.

        unless kwargs['verify'] = False, basic assertions that check the
        integrity are made (e.g., that channel information and whether or not
        data are complex are consistent between header and data).

        Returns
        -------
        frame : VLBIFrameBase instance.
        """
        verify = kwargs.pop('verify', True)
        payload = cls._payload_class.fromdata(data, *args, **kwargs)
        return cls(header, payload, verify)

    def todata(self, data=None):
        return self.payload.todata(data)

    data = property(todata, doc="Decode the payload")

    @property
    def shape(self):
        return self.payload.shape

    @property
    def dtype(self):
        return self.payload.dtype

    @property
    def words(self):
        return np.hstack((np.array(self.header.words), self.payload.words))

    @property
    def size(self):
        return self.header.size + self.payload.size

    def __array__(self):
        return self.payload.data

    def __getitem__(self, item):
        # Header behaves as a dictionary.
        return self.header.__getitem__(item)

    def keys(self):
        return self.header.keys()

    def __contains__(self, key):
        return key in self.header.keys()

    def __getattr__(self, attr):
        try:
            return self.__getattribute__(attr)
        except AttributeError:
            if attr in self.header._properties:
                return getattr(self.header, attr)

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.header == other.header and
                self.payload == other.payload)
