# Atomic Operations

## Introduction
* x++ is a read-modify-write operation
   * Read x into a register
   * Increment register value
   * Write register back into x
   * Effectively { temp = x; temp = temp+1; x = temp; }
* If two threads do x++
   * Each thread has its own temp (say t1 and t2)
   * `{ t1 = x; t1 = t1+1; x = t1;}`
   * `{ t2 = x; t2 = t2+1; x = t2;}`
   * Race condition: the thread that writes to x first wins
   * Whoever wins, x gets incremented only once

## Atomic Functions
* Problem: many threads accessing the same memory location
* Atomic operations ensure that only one thread can access the location
* Grid scope!
atomicOP(x,y)
   * `t1 = *x; // read`
   * `t2 = t1 OP y; // modify`
   * `*a = t2; // write`
* Atomics need to be configured
   * `#include "sm_20_atomic_functions.h“`

## Summary
* Atomic operations ensure operations on a variable cannot be interrupted by a different thread
* CUDA supports several atomic operations
   * `atomicAdd()`
   * `atomicOr()`
   * `atomicMin()`
   * … and others
* Atomics incur a heavy performance penalty
