from collections import namedtuple
from copy import deepcopy

import numpy as np
import ROOT


Bin = namedtuple('Bin', ['content', 'error'])


class Hist1D:
    """Representation of 1D histogram.
    
    Stores NumPy arrays that represent the binning, bin contents, and
    errors.  Under- and overflow bins are supported.
    """
    
    def __init__(self, *args, binning=None, contents=None, errors=None):
        """Initialize from binning or ROOT histogram.
        
        Several ways to initialize Hist1D are supported:
        
        1. From a 1D ROOT histogram.
        This must be the only argument.
        
        2. From an equidistant binning.
        The binning is constructed from three arguments that are
        interpreted as the number of bins (under- and overflows are not
        counted) and the lower and the upper edges.
        
        3. From a binning described by an array.
        The binning is provided via the keyword argument.
        
        4. Blanc histogram.
        If no arguments are given, a blanc histogram is created.  It
        contains no information, but any histogram can be added or
        subtracted from it.  This is useful for creating aggregators.
        
        In variants 2 and 3, histogram contents and errors or contents
        alone can also be set via keyword arguments.  Provided arrays
        may or may not include entries for under- and overflow bins.  In
        the latter case these bins are kept empty.  If bin contents are
        provided but not bin errors, the error in each bin is computed
        as the square root of its content.
        """
        
        if len(args) == 0 and binning is None and contents is None and errors is None:
            # This is a blanc histogram
            self.binning = None
            self.contents = None
            self.errors = None
        
        elif len(args) == 1:
            # Initialize from a ROOT histogram
            if not isinstance(args[0], ROOT.TH1):
                raise TypeError('ROOT histogram expected, got {}.'.format(type(args[0])))
            
            if binning is not None or contents is not None or errors is not None:
                raise RuntimeError(
                    'When initializing from a ROOT histogram, no other arguments are allowed.'
                )
            
            hist = args[0]
            
            if hist.GetDimension() != 1:
                raise RuntimeError('1D histogram is expected.')
            
            numbins = hist.GetNbinsX()
            self.binning = np.zeros(numbins + 1, dtype=np.float64)
            self.contents = np.zeros(numbins + 2, dtype=np.float64)
            self.errors = np.zeros_like(self.contents)
            
            for bin in range(1, numbins + 2):
                self.binning[bin - 1] = hist.GetBinLowEdge(bin)
            
            for bin in range(numbins + 2):
                self.contents[bin] = hist.GetBinContent(bin)
                self.errors[bin] = hist.GetBinError(bin)
        
        elif len(args) in [0, 3]:
            if len(args) == 3:
                # Assume arguments define an equidistant binning
                self.binning = np.linspace(args[1], args[2], num=args[0] + 1)
                
                if binning is not None:
                    raise RuntimeError('Conflicting definitions of binning.')
            
            else:
                if binning is None:
                    raise RuntimeError('Binning must be provided.')
                    
                self.binning = np.asarray(binning, dtype=np.float64)
            
            # With the binning specified, set bin contents and errors
            self.contents = np.zeros(len(self.binning) + 1, dtype=np.float64)
            self.errors = np.zeros_like(self.contents)
            
            if contents is not None:
                if len(contents) == len(self.contents):
                    self.contents[:] = contents
                elif len(contents) == len(self.contents) - 2:
                    # Assume under- and overflows are missing
                    self.contents[1:-1] = contents
                else:
                    raise RuntimeError('Unexpected length of array of bin contentss.')
            
            if errors is not None:
                if len(errors) == len(self.errors):
                    self.errors[:] = errors
                elif len(errors) == len(self.errors) - 2:
                    # Assume under- and overflows are missing
                    self.errors[1:-1] = errors
                else:
                    raise RuntimeError('Unexpected length of array of bin errors.')
                
                if contents is not None and len(errors) != len(contents):
                    raise RuntimeError('Inconsistent arrays of bin contentss and errors.')
            
            elif contents is not None:
                self.errors = np.sqrt(self.contents)
        
        else:
            raise RuntimeError('Not a supported way of initialization.')
    
    
    def __add__(self, other):
        """Add another histogram."""
        
        if isinstance(other, ROOT.TH1):
            other = Hist1D(other)
        
        if self.is_blanc:
            return deepcopy(other)
        elif other.is_blanc:
            return deepcopy(self)
        
        self._check_binning(other)
        
        return Hist1D(
            binning=np.copy(self.binning),
            contents=self.contents + other.contents,
            errors=np.hypot(self.errors, other.errors)
        )
    
    
    def __getitem__(self, index):
        """Return content and error for bin with given index.
        
        Under- and overflow bins have indices 0 and -1.
        """
        
        return Bin(self.contents[index], self.errors[index])
    
    
    def __iadd__(self, other):
        """Add another histogram to this one."""
        
        if isinstance(other, ROOT.TH1):
            other = Hist1D(other)
        
        if other.is_blanc:
            return self
        
        if self.is_blanc:
            self._reset_on_template(other)
        
        self._check_binning(other)
        
        self.contents += other.contents
        self.errors = np.hypot(self.errors, other.errors)
        
        return self
    
    
    def __imul__(self, factor):
        """Multiply contents of this histogram by given factor."""
        
        if self.is_blanc:
            return self
        
        self.contents *= factor
        self.errors *= factor
        
        return self
    
    
    def __isub__(self, other):
        """Subtract another histogram from this one."""
        
        if isinstance(other, ROOT.TH1):
            other = Hist1D(other)
        
        if other.is_blanc:
            return self
        
        if self.is_blanc:
            self._reset_on_template(other)
        
        self._check_binning(other)
        
        self.contents -= other.contents
        self.errors = np.hypot(self.errors, other.errors)
        
        return self
    
    
    def __mul__(self, factor):
        """Scale histogram contents by a factor."""
        
        if self.is_blanc:
            return self
        
        return Hist1D(
            binning=np.copy(self.binning),
            contents=self.contents * factor,
            errors=self.errors * factor
        )
    
    
    __rmul__ = __mul__
    
    
    def __sub__(self, other):
        """Subtract another histogram."""
        
        if isinstance(other, ROOT.TH1):
            other = Hist1D(other)
        
        if self.is_blanc:
            return -1 * deepcopy(other)
        elif other.is_blanc:
            return deepcopy(self)
        
        self._check_binning(other)
        
        return Hist1D(
            binning=np.copy(self.binning),
            contents=self.contents - other.contents,
            errors=np.hypot(self.errors, other.errors)
        )
    
    
    @property
    def is_blanc(self):
        """Check if this is a blanc histogram.
        
        A blanc histogram contains no data.  It can be added to any
        other histogram.
        """
        
        return self.binning is None
    
    
    @property
    def numbins(self):
        """Number of bins in the histogram.
        
        Under- and overflow bins are not counted.
        """
        
        if self.is_blanc:
            raise RuntimeError('Number of bins is not defined for a blanc histogram.')
        
        return len(self.binning) - 1
    
    
    def _check_binning(self, other):
        """Raise an exception is binnings are not identical."""
        
        if not np.array_equal(self.binning, other.binning):
            raise RuntimeError('Binnings of the two histograms being added do not match.')
    
    
    def _reset_on_template(self, other):
        """Copy binning and clear contents."""
        
        self.binning = np.copy(other.binning)
        self.contents = np.zeros(len(self.binning) + 1, dtype=self.binning.dtype)
        self.errors = np.zeros_like(self.contents)
