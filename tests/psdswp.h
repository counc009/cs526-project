#ifndef PAR_SUPPORT_H_
#define PAR_SUPPORT_H_

#include <stdint.h>
#include <pthread.h>

// LLVM type: i8* (i32, ...)
// Variadic arguments are i32 specifying the number of instances of that array
void* createSyncArrays(int numArrays, ...);
// LLVM type: void (i8*, i32, ...)
// Again, variadic arguments are the number of instances
void freeSyncArrays(void* syncArrays, int numArrays, ...);
// LLVM type: void (i8*, i32, i32, i64)
// Currently, only supports passing scalars values up to 64-bits, they must be
// converted into an i64 by bit-extension and bitcasting (as needed)
void produce(void* syncArrays, int toArray, int toRepl, int64_t value);
// LLVM type: i64 (i8*, i32, i32)
int64_t consume(void* syncArrays, int fromArray, int fromRepl);
// LLVM type: i64 (i8*, i8*(i8*))
pthread_t launchStage(void* argument, void*(*func)(void*));
// LLVM type: void (i64)
void waitForStage(pthread_t thread);

#endif // PAR_SUPPORT_H_
