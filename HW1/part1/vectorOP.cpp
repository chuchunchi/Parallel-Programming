#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x;
  __pp_vec_int y, count;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_float nine = _pp_vset_float(9.999999f);
  __pp_vec_int zeroInt = _pp_vset_int(0);
  __pp_mask maskAll, maskIsZero, maskIsNotZero, maskStillCount, maskTooBig;
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // All ones
    maskAll = _pp_init_ones();

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];
    _pp_vload_int(y, exponents + i, maskAll); // y = exponents[i];
    // Set mask according to predicate
    _pp_veq_int(maskIsZero, y, zeroInt, maskAll); // if (y == 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vset_float(result, 1.f, maskIsZero); //   output[i] = 1.f;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotZero = _pp_mask_not(maskIsZero); // } else {

    // Execute instruction ("else" clause) 
    _pp_vmove_float(result, x, maskIsNotZero); // float result = x;
    _pp_vsub_int(count, y, one, maskIsNotZero); // count = y - 1

    _pp_vgt_int(maskStillCount, count, zeroInt, maskIsNotZero); // if(count>0)
    while(_pp_cntbits(maskStillCount)){
      _pp_vmult_float(result, result, x, maskStillCount); // result *= x
      _pp_vsub_int(count, count, one, maskStillCount); // count -= 1

      _pp_vgt_int(maskStillCount, count, zeroInt, maskStillCount); // if(count>0)
    }
    
    _pp_vgt_float(maskTooBig, result, nine, maskIsNotZero); // if(result>9.99f)
    _pp_vset_float(result, 9.999999f, maskTooBig); //   result = 9.99f;

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll); // output[i] = result
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
  }

  return 0.0;
}