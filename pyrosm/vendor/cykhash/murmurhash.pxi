
cdef extern from *:
    """
        #ifndef CYKHASH_MURMURHASH_PXI
        #define CYKHASH_MURMURHASH_PXI
            #include <stdint.h>
            //
            //specializations of https://github.com/aappleby/smhasher/blob/master/src/MurmurHash2.cpp
            //

            const uint32_t SEED = 0xc70f6907UL;
            // 'm' and 'r' are mixing constants generated offline.
            // They're not really 'magic', they just happen to work well.
            const uint32_t M_32 = 0x5bd1e995;
            const int R_32 = 24;
            CYKHASH_INLINE uint32_t murmur2_32to32(uint32_t k){
                // Initialize the hash to a 'random' value
                uint32_t h = SEED ^ 4;

                //handle 4 bytes:
                k *= M_32;
                k ^= k >> R_32;
                k *= M_32;

                h *= M_32;
                h ^= k;

                // Do a few final mixes of the hash to ensure the "last few
                // bytes" are well-incorporated.
                // TODO: really needed, we have no "last few bytes"?
                h ^= h >> 13;
                h *= M_32;
                h ^= h >> 15;
                return h;
            }


            
            #if INTPTR_MAX == INT64_MAX
               // 64-bit
                const uint64_t SEED_64 = 0xc70f6907b8107a18ULL;
                const uint64_t M_64=     0xc6a4a7935bd1e995ULL;
                const int R_64 = 47;
                CYKHASH_INLINE uint32_t murmur2_64to32(uint64_t k){
                      uint64_t h = SEED_64 ^ (8 * M_64);

                      k *= M_64; 
                      k ^= k >> R_64; 
                      k *= M_64; 
                        
                      h ^= k;
                      h *= M_64; 

                      h ^= h >> R_64;
                      h *= M_64;
                      h ^= h >> R_64;

                      // if hash h is good, we just can xor both halfs 
                      // (or take any 32 bit out of h)
                      return (uint32_t)((h>>32)^h);
                }
            #elif INTPTR_MAX == INT32_MAX
                // 32-bit
                // uint64_t mult is slow for 32 bit, so falling back to murmur2_32to32 algorithm
                CYKHASH_INLINE uint32_t murmur2_32_32to32(uint32_t k1, uint32_t k2){
                    // Initialize the hash to a 'random' value
                    uint32_t h = SEED ^ 4;

                    //handle first 4 bytes:
                    k1 *= M_32;
                    k1 ^= k1 >> R_32;
                    k1 *= M_32;

                    h *= M_32;
                    h ^= k1;

                    //handle second 4 bytes:
                    k2 *= M_32;
                    k2 ^= k2 >> R_32;
                    k2 *= M_32;

                    h *= M_32;
                    h ^= k2;

                    // Do a few final mixes of the hash to ensure the "last few
                    // bytes" are well-incorporated.
                    // TODO: really needed, we have no "last few bytes"?
                    h ^= h >> 13;
                    h *= M_32;
                    h ^= h >> 15;
                    return h;
                }

                CYKHASH_INLINE uint32_t murmur2_64to32(uint64_t k){
                      uint32_t k1=(uint32_t)k;
                      uint32_t k2=(uint32_t)(k>>32);

                      return murmur2_32_32to32(k1, k2);
                }
            #else
            #error Unknown pointer size or missing size macros!
            #endif
        #endif
    """
    pass
