
#ifndef _LIFTBASE_H_
#define _LIFTBASE_H_

/** \file

   <b>Copyright and Use</b>

   You may use this source code without limitation and without
   fee as long as you include:

<blockquote>
     This software was written and is copyrighted by Ian Kaplan, Bear
     Products International, www.bearcave.com, 2002.
</blockquote>

   This software is provided "as is", without any warranty or
   claim as to its usefulness.  Anyone who uses this source code
   uses it at their own risk.  Nor is any support provided by
   Ian Kaplan and Bear Products International.

   Please send any bug fixes or suggested source changes to:
<pre>
     iank@bearcave.com
</pre>

  @author Ian Kaplan

  */ 

#include <assert.h>

/**
  This is the base class for simple Lifting Scheme wavelets using
  split, predict, update or update, predict, merge steps.


  Simple lifting scheme wavelets consist of three steps,
  a split/merge step, predict step and an update step:

  <ul>
    <li>
    <p>
    The split step divides the elements in an array so that 
    the even elements are in the first half and the odd
    elements are in the second half.
    </p>
    </li>
    <li>
    <p>
    The merge step is the inverse of the split step.  It takes
    two regions of an array, an odd region and an even region
    and merges them into a new region where an even element 
    alternates with an odd element.
    </p>
    </li>
    <li>
    <p>
    The predict step calculates the difference
    between an odd element and its predicted value based
    on the even elements.  The difference between the
    predicted value and the actual value replaces the
    odd element.
    </p>
    </li>
    <li>
    <p>
    The predict step operates on the odd elements.  The update
    step operates on the even element, replacing them with a
    difference between the predict value and the actual odd element.
    The update step replaces each even element with an average.
    The result of the update step becomes the input to the 
    next recursive step in the wavelet calculation.
    </p>
    </li>

  </ul>

  The split and merge methods are shared by all Lifting Scheme wavelet
  algorithms.  This base class provides the transform and inverse
  transform methods (forwardTrans and inverseTrans).  The predict
  and update methods are abstract and are defined for a particular
  Lifting Scheme wavelet sub-class.

  This is a template version of the lifting scheme base class.  The
  template must be instantiated with an array or an object that acts
  like an array.  Objects that act like arrays define the left hand
  side and right hand side index operators: [].  To allow wavelet
  transforms based on this base class to be used with the wavelet
  packet transform, this class makes public both the forward
  and inverse transforms (forwardTrans and inverseTrans) and
  the forward and inverse transform steps (forwardStep and
  inverseStep).  These "step" functions are used to calculate
  the wavelet packet transform.

  <b>Instantiating the Template</b>

  The liftbase template takes two type arguments:

  <ol>
  <li>
  The type of the array or '[]' operator indexable object.
  </li>
  <li>
  The type of the data element.
  </li>
  </ol>

  The simplest example is a wavelet class derived from an instance of
  the liftbase tempate which takes a double array and has a double
  element type.  This declaration is shown below:

  <pre>
  class Haar : public liftbase<double *, double>
  </pre>

  An object type can be used for the first template argument,
  as long as the object supports the '[]' operator, which returns
  an element whose type is defined by the second argument.  In the
  example below, the packcontainer '[]' operator returns a 
  double.

  <pre>
  class Poly : public liftbase<packcontainer, double>
  </pre>

  <b>References:</b>

  <ul>
  <li>
  <a href="http://www.bearcave.com/misl/misl_tech/wavelets/packet/index.html">
  <i>The Wavelet Packet Transform</i></a> by Ian Kaplan, www.bearcave.com.
  </li>
  <li>
  <a 
  href="http://www.bearcave.com/misl/misl_tech/wavelets/lifting/index.html">
  <i>The Wavelet Lifting Scheme</i></a> by Ian Kaplan, www.bearcave.com.
  This is the parent web page for this Java source code.
  </li>
  <li>
  <i>Ripples in Mathematics: the Discrete Wavelet Transform</i> 
  by Arne Jense and Anders la Cour-Harbo, Springer, 2001
  </li>
  <li>
  <i>Building Your Own Wavelets at Home</i> in <a
  href="http://www.multires.caltech.edu/teaching/courses/waveletcourse/">
  Wavelets in Computer Graphics</a>
  </li>
  </ul>

  \author Ian Kaplan

 */
template <class T, class T_elem >
class liftbase {

public:

  typedef enum { 
  /** "enumeration" for forward wavelet transform */
    forward = 1,
  /** "enumeration" for inverse wavelet transform */
    inverse = 2 
  } transDirection;


  /**
    Split the <i>vec</i> into even and odd elements,
    where the even elements are in the first half
    of the vector and the odd elements are in the
    second half.
   */
  void split( T& vec, int N )
  {
    
    int start = 1;
    int end = N - 1;

    while (start < end) {
      for (int i = start; i < end; i = i + 2) {
	T_elem tmp = vec[i];
	vec[i] = vec[i+1];
	vec[i+1] = tmp;
      }
      start = start + 1;
      end = end - 1;
    }
  }

  /**
    Merge the odd elements from the second half of the N element
    region in the array with the even elements in the first
    half of the N element region.  The result will be the
    combination of the odd and even elements in a region
    of length N.
    
   */
  void merge( T& vec, int N )
  {
    int half = N >> 1;
    int start = half-1;
    int end = half;
    
    while (start > 0) {
      for (int i = start; i < end; i = i + 2) {
	T_elem tmp = vec[i];
	vec[i] = vec[i+1];
	vec[i+1] = tmp;
      }
      start = start - 1;
      end = end + 1;
    }
  }


  /** 
    Predict step, to be defined by the subclass

    @param vec input array
    @param N size of region to act on (from 0..N-1)
    @param direction forward or inverse transform

   */
  virtual void predict( T& vec, int N, transDirection direction ) = 0;

  /**
    Reverse predict step.  

    The predict step applied the high pass filter to the data
    set and places the result in the upper half of the array.
    The reverse predict step applies the high pass filter and
    places the result in the lower half of the array.

    This reverse predict step is only used by wavelet packet
    frequency analysis algorithms.  The default version
    of this algorihtm does nothing.
   */
  virtual void predictRev( T& vec, int N, transDirection direction ) {};
  

  /** 
    Update step, to be defined by the subclass 

    @param vec input array
    @param N size of region to act on (from 0..N-1)
    @param direction forward or inverse transform

  */
  virtual void update( T& vec, int N, transDirection direction ) = 0;


  /**
    Reverse update step
   */
  virtual void updateRev( T& vec, int N, transDirection direction ) {}

public:

  /**
    One step in the forward wavelet transform
   */
  virtual void forwardStep( T& vec, const int n )
  {
    split( vec, n );
    predict( vec, n, forward );
    update( vec, n, forward );
  } // forwardStep

  /**
    Reverse forward transform step.  The result of the high
    pass filter is stored in the lower half of the array
    and the result of the low pass filter is stored in the
    upper half.

    This function should be defined by any subclass that
    is used for wavelet frequency analysis.
   */
  virtual void forwardStepRev( T& vec, const int N )
  {
    assert(false);
  }

  /**
    Simple wavelet Lifting Scheme forward transform

     forwardTrans is passed an indexable object.  The object must
    contain a power of two number of data elements.  Lifting Scheme
    wavelet transforms are calculated in-place and the result is
    returned in the argument array.

    The result of forwardTrans is a set of wavelet coefficients
    ordered by increasing frequency and an approximate average
    of the input data set in vec[0].  The coefficient bands
    follow this element in powers of two (e.g., 1, 2, 4, 8...).

  */
  virtual void forwardTrans( T& vec, const int N )
  {

    for (int n = N; n > 1; n = n >> 1) {
      forwardStep( vec, n );
    }
  } // forwardTrans


  /**
    One inverse wavelet transform step
   */
  virtual void inverseStep( T& vec, const int n )
  {
    update( vec, n, inverse );
    predict( vec, n, inverse );
    merge( vec, n );
  }

  /** 
    Reverse inverse transform step.  Calculate the inverse transform
    from a high pass filter result stored in the lower half of the
    array and a low pass filter result stored in the upper half.

    This function should be defined by any subclass that
    is used for wavelet frequency analysis.
   */
  virtual void inverseStepRev( T& vec, const int n )
  {
    assert( false );
  }


  /**
    Default two step Lifting Scheme inverse wavelet transform

    inverseTrans is passed the result of an ordered wavelet 
    transform, consisting of an average and a set of wavelet
    coefficients.  The inverse transform is calculated
    in-place and the result is returned in the argument array.

   */
  virtual void inverseTrans( T& vec, const int N )
  {

    for (int n = 2; n <= N; n = n << 1) {
      inverseStep( vec, n );
    }
  } // inverseTrans


}; // liftbase

#endif
