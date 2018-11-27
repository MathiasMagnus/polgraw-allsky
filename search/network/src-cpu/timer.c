#define NANO_INV 1000000000L

// Custom include
#include <timer.h>

// C90 includes
#include <stdio.h>          // perror

// C99 includes
#include <stdbool.h>        // bool


struct timespec get_current_time()
{
    struct timespec t;

    bool success = timespec_get(&t, TIME_UTC) == TIME_UTC;

    if (!success)
        perror("Failed to invoke CRT timer function.");

    return t;
}


time_t get_time_difference(struct timespec t0, struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) * NANO_INV + (t1.tv_nsec - t0.tv_nsec);
}
