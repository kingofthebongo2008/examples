
#ifndef _HAAR_H_
#define _HAAR_H_

#include <math.h>

#include "liftbase.h"

/** \file


  The documentation in this file is formatted for doxygen
  (see www.doxygen.org).

<h4>
   Copyright and Use
</h4>

<p>
   You may use this source code without limitation and without
   fee as long as you include:
</p>
<blockquote>
     This software was written and is copyrighted by Ian Kaplan, Bear
     Products International, www.bearcave.com, 2002.
</blockquote>
<p>
   This software is provided "as is", without any warranty or
   claim as to its usefulness.  Anyone who uses this source code
   uses it at their own risk.  Nor is any support provided by
   Ian Kaplan and Bear Products International.
<p>
   Please send any bug fixes or suggested source changes to:
<pre>
     iank@bearcave.com
</pre>

  @author Ian Kaplan

 */

/**
  Haar (flat line) wavelet.

  As with all Lifting scheme wavelet transform functions, the
  first stage of a transform step is the split stage.  The
  split step moves the even element to the first half of an
  N element region and the odd elements to the second half of the N
  element region.

  The Lifting Scheme version of the Haar transform uses a wavelet
  function (predict stage) that "predicts" that an odd element will
  have the same value as it preceeding even element.  Stated another
  way, the odd element is "predicted" to be on a flat (zero slope
  line) shared with the even point.  The difference between this
  "prediction" and the actual odd value replaces the odd element.

  The wavelet scaling function (a.k.a. smoothing function) used
  in the update stage calculates the average between an even and
  an odd element.

  The merge stage at the end of the inverse transform interleaves
  odd and even elements from the two halves of the array
  (e.g., ordering them even<sub>0</sub>, odd<sub>0</sub>,
  even<sub>1</sub>, odd<sub>1</sub>, ...)

  This is a template version of the Haar wavelet.  The template must
  be instantiated with an array or an object that acts like an array.
  Objects that act like arrays define the left hand side and right
  hand side index operators: [].

  See www.bearcave.com for more information on wavelets and the
  wavelet lifting scheme.

  \author Ian Kaplan

 */
template <class T>
class haar : public liftbase<T, double> {

public:
  /**
    Haar predict step
   */
  void predict( T& vec, int N, transDirection direction )
  {
    int half = N >> 1;

    for (int i = 0; i < half; i++) {
      double predictVal = vec[i];
      int j = i + half;

      if (direction == forward) {
	vec[j] = vec[j] - predictVal;
      }
      else if (direction == inverse) {
	vec[j] = vec[j] + predictVal;
      }
      else {
	printf("haar::predict: bad direction value\n");
      }
    }
  }


  /**
    Update step of the Haar wavelet transform.

    The wavelet transform calculates a set of detail or
    difference coefficients in the predict step.  These
    are stored in the upper half of the array.  The update step
    calculates an average from the even-odd element pairs.
    The averages will replace the even elements in the 
    lower half of the array.

    The Haar wavelet calculation used in the Lifting Scheme
    is

    <pre>
       d<sub>j+1, i</sub> = odd<sub>j+1, i</sub> = odd<sub>j, i</sub> - even<sub>j, i</sub>
       a<sub>j+1, i</sub> = even<sub>j, i</sub> = (even<sub>j, i</sub> + odd<sub>j, i</sub>)/2
    </pre>

    Note that the Lifting Scheme uses an in-place algorithm.  The odd
    elements have been replaced by the detail coefficients in the
    predict step.  With a little algebra we can substitute the
    coefficient calculation into the average calculation, which
    gives us

    <pre>
       a<sub>j+1, i</sub> = even<sub>j, i</sub> = even<sub>j, i</sub> + (odd<sub>j, i</sub>/2)
    </pre>
   */
  void update( T& vec, int N, transDirection direction )
  {
    int half = N >> 1;

    for (int i = 0; i < half; i++) {
      int j = i + half;
      double updateVal = vec[j] / 2.0;

      if (direction == forward) {
	vec[i] = vec[i] + updateVal;
      }
      else if (direction == inverse) {
	vec[i] = vec[i] - updateVal;
      }
      else {
	printf("update: bad direction value\n");
      }
    }
  }

  /**
    The normalization step assures that each step of the wavelet
    transform has the constant "energy" where energy is defined as

    <pre>
    double energy = 0.0;
    for (int n = 0; n < N; n++) {
       energy = energy + (a[i] * a[i]);
    }
    </pre>

    See 5.2.1 of <i>Ripples in Mathematics</i> by Jensen
    and la Cour-Harbo

    The most common implementation of the Haar transform leaves out
    the normalization step, since it does not make much of a
    difference in many cases.  However, in the case of the wavelet
    packet transform, many of the cost functions are squares, so
    normalization produces smaller wavelet values (although the
    scaling function values are larger).  This may lead to a better
    wavelet packet result (e.g., a few large values and lots of small
    values).

    Normalization does have the disadvantage of destroying the
    averaging property of the Haar wavelet algorithm.  That is, the
    final scaling factor is no longer the mean of the time series.

   */
  void normalize( T& vec, int N, transDirection direction )
  {
    const double sqrt2 = sqrt( 2.0 );
    int half = N >> 1;

    for (int i = 0; i < half; i++) {
      int j = i + half;

      if (direction == forward) {
	vec[i] = sqrt2 * vec[i];
	vec[j] = vec[j]/sqrt2;
      }
      else if (direction == inverse) {
	vec[i] = vec[i]/sqrt2;
	vec[j] = sqrt2 * vec[j];
      }
      else {
	printf("normalize: bad direction value\n");
      }
    } // for
  } // normalize

  /**
    One inverse wavelet transform step, with normalization
   */
  void inverseStep( T& vec, const int n )
  {
    normalize( vec, n, inverse );
    update( vec, n, inverse );
    predict( vec, n, inverse );
    merge( vec, n );
  }  // inverseStep

  /**
    One step in the forward wavelet transform, with normalization
   */
  void forwardStep( T& vec, const int n )
  {
    split( vec, n );
    predict( vec, n, forward );
    update( vec, n, forward );
    normalize( vec, n, forward );
  } // forwardStep


}; // haar

#endif
