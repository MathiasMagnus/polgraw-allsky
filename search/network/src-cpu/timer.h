#ifndef __TIMER_H__
#define __TIMER_H__

// C11 standard includes
#include <time.h>           // struct timespec, time_t

/// <summary>Obtain the current time through C11 timers.</summary>
///
struct timespec get_current_time();

/// <summary>Obtain the difference between two time points in seconds.</summary>
///
time_t get_time_difference(struct timespec t0, struct timespec t1);

#endif
