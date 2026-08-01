

cdef extern from *:
    """ 
        #ifndef CYKHASH_MEMORY_PXI
        #define CYKHASH_MEMORY_PXI
        #include <Python.h>

        // cykhash should report usage to tracemalloc
        #if PY_VERSION_HEX >= 0x03060000
        #include <pymem.h>
        #if PY_VERSION_HEX < 0x03070000
        #define PyTraceMalloc_Track _PyTraceMalloc_Track
        #define PyTraceMalloc_Untrack _PyTraceMalloc_Untrack
        #endif
        #else
        #define PyTraceMalloc_Track(...)
        #define PyTraceMalloc_Untrack(...)
        #endif


        static const int CYKHASH_TRACE_DOMAIN = 414141;
        CYKHASH_INLINE void *cykhash_traced_malloc(size_t size){
            void * ptr = malloc(size);
            if(ptr!=NULL){
                PyTraceMalloc_Track(CYKHASH_TRACE_DOMAIN, (uintptr_t)ptr, size);
            }
            return ptr;
        }

        CYKHASH_INLINE void *cykhash_traced_calloc(size_t num, size_t size){
            void * ptr = calloc(num, size);
            if(ptr!=NULL){
                PyTraceMalloc_Track(CYKHASH_TRACE_DOMAIN, (uintptr_t)ptr, num*size);
            }
            return ptr;
        }

        CYKHASH_INLINE void *cykhash_traced_realloc(void* old_ptr, size_t size){
            void * ptr = realloc(old_ptr, size);
            if(ptr!=NULL){
                if(old_ptr != ptr){
                    PyTraceMalloc_Untrack(CYKHASH_TRACE_DOMAIN, (uintptr_t)old_ptr);
                }
                PyTraceMalloc_Track(CYKHASH_TRACE_DOMAIN, (uintptr_t)ptr, size);
            }
            return ptr;
        }

        CYKHASH_INLINE void cykhash_traced_free(void* ptr){
            if(ptr!=NULL){
                PyTraceMalloc_Untrack(CYKHASH_TRACE_DOMAIN, (uintptr_t)ptr);
            }
            free(ptr);
        }


        #define CYKHASH_MALLOC cykhash_traced_malloc
        #define CYKHASH_REALLOC cykhash_traced_realloc
        #define CYKHASH_CALLOC cykhash_traced_calloc
        #define CYKHASH_FREE cykhash_traced_free
      
        #endif
    """
    const int CYKHASH_TRACE_DOMAIN
    void *cykhash_traced_malloc(size_t size)
    void *cykhash_traced_calloc(size_t num, size_t size)
    void *cykhash_traced_realloc(void* old_ptr, size_t size)
    void cykhash_traced_free(void* ptr)

