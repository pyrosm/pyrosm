"""
Template for maps

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""


from cpython.ref cimport Py_INCREF,Py_DECREF





cdef class Int64toInt64Map:

    @classmethod
    def fromkeys(cls, iterable, value):
        return Int64toInt64Map(((key, value) for key in iterable))

    def __cinit__(self, iterable=None, *, number_of_elements_hint=None):
        """
        number_of_elements_hint - number of elements without the need of reallocation.
        """
        self.table = kh_init_int64toint64map()
        if number_of_elements_hint is not None:
            kh_resize_int64toint64map(self.table, element_n_to_bucket_n(number_of_elements_hint))
        cdef int64_t key
        cdef int64_t val
        if iterable is not None:
            for key, val in iterable:
                    self.cput(key, val)

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_int64toint64map(self.table)
            self.table = NULL

    cpdef void discard(self, int64_t key) except *:
        cdef khint_t k
        k = kh_get_int64toint64map(self.table, key)
        if k != self.table.n_buckets:
            kh_del_int64toint64map(self.table, k)

    cdef bint contains(self, int64_t key) except *:
        cdef khint_t k
        k = kh_get_int64toint64map(self.table, key)
        return k != self.table.n_buckets

    def __contains__(self, int64_t key):
        return self.contains(key)


    def __len__(self):
        return self.size()
  
    cdef khint_t size(self):
        return self.table.size

    cpdef void cput(self, int64_t key, int64_t val) except *:
        cdef:
            khint_t k
            int ret = 0

        k = kh_put_int64toint64map(self.table, key, &ret)
        self.table.keys[k] = key
        self.table.vals[k] = val

    def __setitem__(self, key, val):
        self.cput(key, val)

    cpdef int64_t cget(self, int64_t key) except *:
        k = kh_get_int64toint64map(self.table, key)
        if k != self.table.n_buckets:
            return self.table.vals[k]
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        return self.cget(key)
        


    cdef Int64toInt64MapIterator get_iter(self, int view_type):
        return Int64toInt64MapIterator(self, view_type)

    def clear(self):
        cdef Int64toInt64Map tmp=Int64toInt64Map()
        swap_int64toint64map(self, tmp)

    def copy(self):
        return copy_int64toint64map(self)

    def update(self, other):
        if isinstance(other, Int64toInt64Map):
            update_int64toint64map(self, other)
            return
        for key,val in other:
            self[key]=val

    def setdefault(self, key, default):
        try:
            return self[key]
        except KeyError:
            self[key]=default
            return default

    def get(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("get() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("get() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("get() takes no keyword arguments")
        key = args[0]
        try:
            return self[key]
        except KeyError:
            if len(args)==1:
                return None
            return args[1]

    def pop(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("pop() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("pop() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("pop() takes no keyword arguments")
        key = args[0]
        try:
            val = self[key]
        except KeyError as e:
            if len(args)==1:
                raise e from None
            return args[1]
        del self[key]
        return val

    def popitem(self):
        if self.size()== 0:
            raise KeyError("popitem(): dictionary is empty")
        key = next(iter(self))
        val = self.pop(key)
        return (key, val)

    def keys(self):
        return Int64toInt64MapView(self, 0)

    def values(self):
        return Int64toInt64MapView(self, 1)

    def items(self):
        return Int64toInt64MapView(self, 2)

    def __iter__(self):
        return iter(self.keys())

    def __delitem__(self, key):
        cdef size_t old=self.size()
        self.discard(key)
        if old==self.size():
            raise KeyError(key)

    def __eq__(self, other):
        return are_equal_int64toint64map(self,other)


### Iterator:
cdef class Int64toInt64MapIterator:

    cdef void __move(self) except *:
        while self.it<self.parent.table.n_buckets and not kh_exist_int64toint64map(self.parent.table, self.it):
              self.it+=1       

    cdef bint has_next(self) except *:
        self.__move()
        return self.it < self.parent.table.n_buckets
      
    # doesn't work if there was change between last has_next() and next()       
    cdef int64toint64_key_val_pair next(self) except *:
        cdef int64toint64_key_val_pair result 
        result.key = self.parent.table.keys[self.it]
        result.val = self.parent.table.vals[self.it]
        self.it+=1#ensure at least one move!
        return result

    def __cinit__(self, Int64toInt64Map parent, view_type):
        self.parent = parent
        self.view_type = view_type
        #search the start:
        self.it = 0
        self.__move()

    def __next__(self):
        cdef int64toint64_key_val_pair pair
        if self.has_next():
            pair=self.next()

            if self.view_type == 0:           # keys
                return pair.key
            if self.view_type == 1:           # vals
                return pair.val
            else:                        # items
                return (pair.key, pair.val)
        else:
            raise StopIteration


cdef class Int64toInt64MapView:
    cdef Int64toInt64MapIterator get_iter(self):
        return Int64toInt64MapIterator(self.parent, self.view_type)  

    def __cinit__(self, Int64toInt64Map parent, view_type):
        self.parent = parent
        self.view_type = view_type

    def __iter__(self):
        return self.get_iter()

    def __len__(self):
        return self.parent.size()

    def __contains__(self, x):
        for y in self:
            if x==y:
                return True
        return False

##########################      Utils:
cpdef Int64toInt64Map Int64toInt64Map_from_buffers(int64_t[:] keys, int64_t[:] vals, double size_hint=0.0):
    cdef Py_ssize_t n = len(keys)
    cdef Py_ssize_t b = len(vals)
    if b < n:
        n = b
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Int64toInt64Map(number_of_elements_hint=at_least_needed)
    cdef Py_ssize_t i
    for i in range(n):
        res.cput(keys[i], vals[i])
    return res

cdef int64_t DEFAULT_VALUE_int64toint64 = 0
cpdef size_t Int64toInt64Map_to(Int64toInt64Map map, int64_t[:] keys, int64_t[:] vals, bint stop_at_unknown=True, int64_t default_value=DEFAULT_VALUE_int64toint64) except *:
    """returns number of found keys"""
    if map is None:
        raise TypeError("'NoneType' is not a map")
    cdef size_t n = len(keys)
    if n != len(vals):
        raise ValueError("Different lengths of keys and vals arrays")
    cdef size_t i
    cdef khint_t k
    cdef size_t res = 0
    for i in range(n):
        k = kh_get_int64toint64map(map.table, keys[i])
        if k != map.table.n_buckets:
            vals[i] = map.table.vals[k]
            res += 1
        else:
            vals[i] = default_value
            if stop_at_unknown:
                return res
    return res

cpdef void swap_int64toint64map(Int64toInt64Map a, Int64toInt64Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef kh_int64toint64map_t *tmp=a.table
    a.table=b.table
    b.table=tmp


cpdef Int64toInt64Map copy_int64toint64map(Int64toInt64Map s):
    if s is None:
        return None
    cdef Int64toInt64Map result = Int64toInt64Map(number_of_elements_hint=s.size())
    cdef Int64toInt64MapIterator it=s.get_iter(2)
    cdef int64toint64_key_val_pair p
    while it.has_next():
        p = it.next()
        result.cput(p.key, p.val)
    return result


cpdef bint are_equal_int64toint64map(Int64toInt64Map a, Int64toInt64Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    if a.size()!=b.size():
        return False
    cdef Int64toInt64MapIterator it=a.get_iter(2)
    cdef int64toint64_key_val_pair p
    while it.has_next():
        p = it.next()
        if not b.contains(p.key):
            return False
    return True


cpdef void update_int64toint64map(Int64toInt64Map a, Int64toInt64Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    cdef Int64toInt64MapIterator it=b.get_iter(2)
    cdef int64toint64_key_val_pair p
    while it.has_next():
        p = it.next()
        a.cput(p.key, p.val)


cdef class Int64toFloat64Map:

    @classmethod
    def fromkeys(cls, iterable, value):
        return Int64toFloat64Map(((key, value) for key in iterable))

    def __cinit__(self, iterable=None, *, number_of_elements_hint=None):
        """
        number_of_elements_hint - number of elements without the need of reallocation.
        """
        self.table = kh_init_int64tofloat64map()
        if number_of_elements_hint is not None:
            kh_resize_int64tofloat64map(self.table, element_n_to_bucket_n(number_of_elements_hint))
        cdef int64_t key
        cdef float64_t val
        if iterable is not None:
            for key, val in iterable:
                    self.cput(key, val)

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_int64tofloat64map(self.table)
            self.table = NULL

    cpdef void discard(self, int64_t key) except *:
        cdef khint_t k
        k = kh_get_int64tofloat64map(self.table, key)
        if k != self.table.n_buckets:
            kh_del_int64tofloat64map(self.table, k)

    cdef bint contains(self, int64_t key) except *:
        cdef khint_t k
        k = kh_get_int64tofloat64map(self.table, key)
        return k != self.table.n_buckets

    def __contains__(self, int64_t key):
        return self.contains(key)


    def __len__(self):
        return self.size()
  
    cdef khint_t size(self):
        return self.table.size

    cpdef void cput(self, int64_t key, float64_t val) except *:
        cdef:
            khint_t k
            int ret = 0

        k = kh_put_int64tofloat64map(self.table, key, &ret)
        self.table.keys[k] = key
        self.table.vals[k] = val

    def __setitem__(self, key, val):
        self.cput(key, val)

    cpdef float64_t cget(self, int64_t key) except *:
        k = kh_get_int64tofloat64map(self.table, key)
        if k != self.table.n_buckets:
            return self.table.vals[k]
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        return self.cget(key)
        


    cdef Int64toFloat64MapIterator get_iter(self, int view_type):
        return Int64toFloat64MapIterator(self, view_type)

    def clear(self):
        cdef Int64toFloat64Map tmp=Int64toFloat64Map()
        swap_int64tofloat64map(self, tmp)

    def copy(self):
        return copy_int64tofloat64map(self)

    def update(self, other):
        if isinstance(other, Int64toFloat64Map):
            update_int64tofloat64map(self, other)
            return
        for key,val in other:
            self[key]=val

    def setdefault(self, key, default):
        try:
            return self[key]
        except KeyError:
            self[key]=default
            return default

    def get(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("get() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("get() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("get() takes no keyword arguments")
        key = args[0]
        try:
            return self[key]
        except KeyError:
            if len(args)==1:
                return None
            return args[1]

    def pop(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("pop() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("pop() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("pop() takes no keyword arguments")
        key = args[0]
        try:
            val = self[key]
        except KeyError as e:
            if len(args)==1:
                raise e from None
            return args[1]
        del self[key]
        return val

    def popitem(self):
        if self.size()== 0:
            raise KeyError("popitem(): dictionary is empty")
        key = next(iter(self))
        val = self.pop(key)
        return (key, val)

    def keys(self):
        return Int64toFloat64MapView(self, 0)

    def values(self):
        return Int64toFloat64MapView(self, 1)

    def items(self):
        return Int64toFloat64MapView(self, 2)

    def __iter__(self):
        return iter(self.keys())

    def __delitem__(self, key):
        cdef size_t old=self.size()
        self.discard(key)
        if old==self.size():
            raise KeyError(key)

    def __eq__(self, other):
        return are_equal_int64tofloat64map(self,other)


### Iterator:
cdef class Int64toFloat64MapIterator:

    cdef void __move(self) except *:
        while self.it<self.parent.table.n_buckets and not kh_exist_int64tofloat64map(self.parent.table, self.it):
              self.it+=1       

    cdef bint has_next(self) except *:
        self.__move()
        return self.it < self.parent.table.n_buckets
      
    # doesn't work if there was change between last has_next() and next()       
    cdef int64tofloat64_key_val_pair next(self) except *:
        cdef int64tofloat64_key_val_pair result 
        result.key = self.parent.table.keys[self.it]
        result.val = self.parent.table.vals[self.it]
        self.it+=1#ensure at least one move!
        return result

    def __cinit__(self, Int64toFloat64Map parent, view_type):
        self.parent = parent
        self.view_type = view_type
        #search the start:
        self.it = 0
        self.__move()

    def __next__(self):
        cdef int64tofloat64_key_val_pair pair
        if self.has_next():
            pair=self.next()

            if self.view_type == 0:           # keys
                return pair.key
            if self.view_type == 1:           # vals
                return pair.val
            else:                        # items
                return (pair.key, pair.val)
        else:
            raise StopIteration


cdef class Int64toFloat64MapView:
    cdef Int64toFloat64MapIterator get_iter(self):
        return Int64toFloat64MapIterator(self.parent, self.view_type)  

    def __cinit__(self, Int64toFloat64Map parent, view_type):
        self.parent = parent
        self.view_type = view_type

    def __iter__(self):
        return self.get_iter()

    def __len__(self):
        return self.parent.size()

    def __contains__(self, x):
        for y in self:
            if x==y:
                return True
        return False

##########################      Utils:
cpdef Int64toFloat64Map Int64toFloat64Map_from_buffers(int64_t[:] keys, float64_t[:] vals, double size_hint=0.0):
    cdef Py_ssize_t n = len(keys)
    cdef Py_ssize_t b = len(vals)
    if b < n:
        n = b
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Int64toFloat64Map(number_of_elements_hint=at_least_needed)
    cdef Py_ssize_t i
    for i in range(n):
        res.cput(keys[i], vals[i])
    return res

cdef float64_t DEFAULT_VALUE_int64tofloat64 = float("nan")
cpdef size_t Int64toFloat64Map_to(Int64toFloat64Map map, int64_t[:] keys, float64_t[:] vals, bint stop_at_unknown=True, float64_t default_value=DEFAULT_VALUE_int64tofloat64) except *:
    """returns number of found keys"""
    if map is None:
        raise TypeError("'NoneType' is not a map")
    cdef size_t n = len(keys)
    if n != len(vals):
        raise ValueError("Different lengths of keys and vals arrays")
    cdef size_t i
    cdef khint_t k
    cdef size_t res = 0
    for i in range(n):
        k = kh_get_int64tofloat64map(map.table, keys[i])
        if k != map.table.n_buckets:
            vals[i] = map.table.vals[k]
            res += 1
        else:
            vals[i] = default_value
            if stop_at_unknown:
                return res
    return res

cpdef void swap_int64tofloat64map(Int64toFloat64Map a, Int64toFloat64Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef kh_int64tofloat64map_t *tmp=a.table
    a.table=b.table
    b.table=tmp


cpdef Int64toFloat64Map copy_int64tofloat64map(Int64toFloat64Map s):
    if s is None:
        return None
    cdef Int64toFloat64Map result = Int64toFloat64Map(number_of_elements_hint=s.size())
    cdef Int64toFloat64MapIterator it=s.get_iter(2)
    cdef int64tofloat64_key_val_pair p
    while it.has_next():
        p = it.next()
        result.cput(p.key, p.val)
    return result


cpdef bint are_equal_int64tofloat64map(Int64toFloat64Map a, Int64toFloat64Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    if a.size()!=b.size():
        return False
    cdef Int64toFloat64MapIterator it=a.get_iter(2)
    cdef int64tofloat64_key_val_pair p
    while it.has_next():
        p = it.next()
        if not b.contains(p.key):
            return False
    return True


cpdef void update_int64tofloat64map(Int64toFloat64Map a, Int64toFloat64Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    cdef Int64toFloat64MapIterator it=b.get_iter(2)
    cdef int64tofloat64_key_val_pair p
    while it.has_next():
        p = it.next()
        a.cput(p.key, p.val)


cdef class Float64toInt64Map:

    @classmethod
    def fromkeys(cls, iterable, value):
        return Float64toInt64Map(((key, value) for key in iterable))

    def __cinit__(self, iterable=None, *, number_of_elements_hint=None):
        """
        number_of_elements_hint - number of elements without the need of reallocation.
        """
        self.table = kh_init_float64toint64map()
        if number_of_elements_hint is not None:
            kh_resize_float64toint64map(self.table, element_n_to_bucket_n(number_of_elements_hint))
        cdef float64_t key
        cdef int64_t val
        if iterable is not None:
            for key, val in iterable:
                    self.cput(key, val)

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_float64toint64map(self.table)
            self.table = NULL

    cpdef void discard(self, float64_t key) except *:
        cdef khint_t k
        k = kh_get_float64toint64map(self.table, key)
        if k != self.table.n_buckets:
            kh_del_float64toint64map(self.table, k)

    cdef bint contains(self, float64_t key) except *:
        cdef khint_t k
        k = kh_get_float64toint64map(self.table, key)
        return k != self.table.n_buckets

    def __contains__(self, float64_t key):
        return self.contains(key)


    def __len__(self):
        return self.size()
  
    cdef khint_t size(self):
        return self.table.size

    cpdef void cput(self, float64_t key, int64_t val) except *:
        cdef:
            khint_t k
            int ret = 0

        k = kh_put_float64toint64map(self.table, key, &ret)
        self.table.keys[k] = key
        self.table.vals[k] = val

    def __setitem__(self, key, val):
        self.cput(key, val)

    cpdef int64_t cget(self, float64_t key) except *:
        k = kh_get_float64toint64map(self.table, key)
        if k != self.table.n_buckets:
            return self.table.vals[k]
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        return self.cget(key)
        


    cdef Float64toInt64MapIterator get_iter(self, int view_type):
        return Float64toInt64MapIterator(self, view_type)

    def clear(self):
        cdef Float64toInt64Map tmp=Float64toInt64Map()
        swap_float64toint64map(self, tmp)

    def copy(self):
        return copy_float64toint64map(self)

    def update(self, other):
        if isinstance(other, Float64toInt64Map):
            update_float64toint64map(self, other)
            return
        for key,val in other:
            self[key]=val

    def setdefault(self, key, default):
        try:
            return self[key]
        except KeyError:
            self[key]=default
            return default

    def get(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("get() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("get() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("get() takes no keyword arguments")
        key = args[0]
        try:
            return self[key]
        except KeyError:
            if len(args)==1:
                return None
            return args[1]

    def pop(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("pop() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("pop() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("pop() takes no keyword arguments")
        key = args[0]
        try:
            val = self[key]
        except KeyError as e:
            if len(args)==1:
                raise e from None
            return args[1]
        del self[key]
        return val

    def popitem(self):
        if self.size()== 0:
            raise KeyError("popitem(): dictionary is empty")
        key = next(iter(self))
        val = self.pop(key)
        return (key, val)

    def keys(self):
        return Float64toInt64MapView(self, 0)

    def values(self):
        return Float64toInt64MapView(self, 1)

    def items(self):
        return Float64toInt64MapView(self, 2)

    def __iter__(self):
        return iter(self.keys())

    def __delitem__(self, key):
        cdef size_t old=self.size()
        self.discard(key)
        if old==self.size():
            raise KeyError(key)

    def __eq__(self, other):
        return are_equal_float64toint64map(self,other)


### Iterator:
cdef class Float64toInt64MapIterator:

    cdef void __move(self) except *:
        while self.it<self.parent.table.n_buckets and not kh_exist_float64toint64map(self.parent.table, self.it):
              self.it+=1       

    cdef bint has_next(self) except *:
        self.__move()
        return self.it < self.parent.table.n_buckets
      
    # doesn't work if there was change between last has_next() and next()       
    cdef float64toint64_key_val_pair next(self) except *:
        cdef float64toint64_key_val_pair result 
        result.key = self.parent.table.keys[self.it]
        result.val = self.parent.table.vals[self.it]
        self.it+=1#ensure at least one move!
        return result

    def __cinit__(self, Float64toInt64Map parent, view_type):
        self.parent = parent
        self.view_type = view_type
        #search the start:
        self.it = 0
        self.__move()

    def __next__(self):
        cdef float64toint64_key_val_pair pair
        if self.has_next():
            pair=self.next()

            if self.view_type == 0:           # keys
                return pair.key
            if self.view_type == 1:           # vals
                return pair.val
            else:                        # items
                return (pair.key, pair.val)
        else:
            raise StopIteration


cdef class Float64toInt64MapView:
    cdef Float64toInt64MapIterator get_iter(self):
        return Float64toInt64MapIterator(self.parent, self.view_type)  

    def __cinit__(self, Float64toInt64Map parent, view_type):
        self.parent = parent
        self.view_type = view_type

    def __iter__(self):
        return self.get_iter()

    def __len__(self):
        return self.parent.size()

    def __contains__(self, x):
        for y in self:
            if x==y:
                return True
        return False

##########################      Utils:
cpdef Float64toInt64Map Float64toInt64Map_from_buffers(float64_t[:] keys, int64_t[:] vals, double size_hint=0.0):
    cdef Py_ssize_t n = len(keys)
    cdef Py_ssize_t b = len(vals)
    if b < n:
        n = b
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Float64toInt64Map(number_of_elements_hint=at_least_needed)
    cdef Py_ssize_t i
    for i in range(n):
        res.cput(keys[i], vals[i])
    return res

cdef int64_t DEFAULT_VALUE_float64toint64 = 0
cpdef size_t Float64toInt64Map_to(Float64toInt64Map map, float64_t[:] keys, int64_t[:] vals, bint stop_at_unknown=True, int64_t default_value=DEFAULT_VALUE_float64toint64) except *:
    """returns number of found keys"""
    if map is None:
        raise TypeError("'NoneType' is not a map")
    cdef size_t n = len(keys)
    if n != len(vals):
        raise ValueError("Different lengths of keys and vals arrays")
    cdef size_t i
    cdef khint_t k
    cdef size_t res = 0
    for i in range(n):
        k = kh_get_float64toint64map(map.table, keys[i])
        if k != map.table.n_buckets:
            vals[i] = map.table.vals[k]
            res += 1
        else:
            vals[i] = default_value
            if stop_at_unknown:
                return res
    return res

cpdef void swap_float64toint64map(Float64toInt64Map a, Float64toInt64Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef kh_float64toint64map_t *tmp=a.table
    a.table=b.table
    b.table=tmp


cpdef Float64toInt64Map copy_float64toint64map(Float64toInt64Map s):
    if s is None:
        return None
    cdef Float64toInt64Map result = Float64toInt64Map(number_of_elements_hint=s.size())
    cdef Float64toInt64MapIterator it=s.get_iter(2)
    cdef float64toint64_key_val_pair p
    while it.has_next():
        p = it.next()
        result.cput(p.key, p.val)
    return result


cpdef bint are_equal_float64toint64map(Float64toInt64Map a, Float64toInt64Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    if a.size()!=b.size():
        return False
    cdef Float64toInt64MapIterator it=a.get_iter(2)
    cdef float64toint64_key_val_pair p
    while it.has_next():
        p = it.next()
        if not b.contains(p.key):
            return False
    return True


cpdef void update_float64toint64map(Float64toInt64Map a, Float64toInt64Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    cdef Float64toInt64MapIterator it=b.get_iter(2)
    cdef float64toint64_key_val_pair p
    while it.has_next():
        p = it.next()
        a.cput(p.key, p.val)


cdef class Float64toFloat64Map:

    @classmethod
    def fromkeys(cls, iterable, value):
        return Float64toFloat64Map(((key, value) for key in iterable))

    def __cinit__(self, iterable=None, *, number_of_elements_hint=None):
        """
        number_of_elements_hint - number of elements without the need of reallocation.
        """
        self.table = kh_init_float64tofloat64map()
        if number_of_elements_hint is not None:
            kh_resize_float64tofloat64map(self.table, element_n_to_bucket_n(number_of_elements_hint))
        cdef float64_t key
        cdef float64_t val
        if iterable is not None:
            for key, val in iterable:
                    self.cput(key, val)

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_float64tofloat64map(self.table)
            self.table = NULL

    cpdef void discard(self, float64_t key) except *:
        cdef khint_t k
        k = kh_get_float64tofloat64map(self.table, key)
        if k != self.table.n_buckets:
            kh_del_float64tofloat64map(self.table, k)

    cdef bint contains(self, float64_t key) except *:
        cdef khint_t k
        k = kh_get_float64tofloat64map(self.table, key)
        return k != self.table.n_buckets

    def __contains__(self, float64_t key):
        return self.contains(key)


    def __len__(self):
        return self.size()
  
    cdef khint_t size(self):
        return self.table.size

    cpdef void cput(self, float64_t key, float64_t val) except *:
        cdef:
            khint_t k
            int ret = 0

        k = kh_put_float64tofloat64map(self.table, key, &ret)
        self.table.keys[k] = key
        self.table.vals[k] = val

    def __setitem__(self, key, val):
        self.cput(key, val)

    cpdef float64_t cget(self, float64_t key) except *:
        k = kh_get_float64tofloat64map(self.table, key)
        if k != self.table.n_buckets:
            return self.table.vals[k]
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        return self.cget(key)
        


    cdef Float64toFloat64MapIterator get_iter(self, int view_type):
        return Float64toFloat64MapIterator(self, view_type)

    def clear(self):
        cdef Float64toFloat64Map tmp=Float64toFloat64Map()
        swap_float64tofloat64map(self, tmp)

    def copy(self):
        return copy_float64tofloat64map(self)

    def update(self, other):
        if isinstance(other, Float64toFloat64Map):
            update_float64tofloat64map(self, other)
            return
        for key,val in other:
            self[key]=val

    def setdefault(self, key, default):
        try:
            return self[key]
        except KeyError:
            self[key]=default
            return default

    def get(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("get() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("get() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("get() takes no keyword arguments")
        key = args[0]
        try:
            return self[key]
        except KeyError:
            if len(args)==1:
                return None
            return args[1]

    def pop(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("pop() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("pop() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("pop() takes no keyword arguments")
        key = args[0]
        try:
            val = self[key]
        except KeyError as e:
            if len(args)==1:
                raise e from None
            return args[1]
        del self[key]
        return val

    def popitem(self):
        if self.size()== 0:
            raise KeyError("popitem(): dictionary is empty")
        key = next(iter(self))
        val = self.pop(key)
        return (key, val)

    def keys(self):
        return Float64toFloat64MapView(self, 0)

    def values(self):
        return Float64toFloat64MapView(self, 1)

    def items(self):
        return Float64toFloat64MapView(self, 2)

    def __iter__(self):
        return iter(self.keys())

    def __delitem__(self, key):
        cdef size_t old=self.size()
        self.discard(key)
        if old==self.size():
            raise KeyError(key)

    def __eq__(self, other):
        return are_equal_float64tofloat64map(self,other)


### Iterator:
cdef class Float64toFloat64MapIterator:

    cdef void __move(self) except *:
        while self.it<self.parent.table.n_buckets and not kh_exist_float64tofloat64map(self.parent.table, self.it):
              self.it+=1       

    cdef bint has_next(self) except *:
        self.__move()
        return self.it < self.parent.table.n_buckets
      
    # doesn't work if there was change between last has_next() and next()       
    cdef float64tofloat64_key_val_pair next(self) except *:
        cdef float64tofloat64_key_val_pair result 
        result.key = self.parent.table.keys[self.it]
        result.val = self.parent.table.vals[self.it]
        self.it+=1#ensure at least one move!
        return result

    def __cinit__(self, Float64toFloat64Map parent, view_type):
        self.parent = parent
        self.view_type = view_type
        #search the start:
        self.it = 0
        self.__move()

    def __next__(self):
        cdef float64tofloat64_key_val_pair pair
        if self.has_next():
            pair=self.next()

            if self.view_type == 0:           # keys
                return pair.key
            if self.view_type == 1:           # vals
                return pair.val
            else:                        # items
                return (pair.key, pair.val)
        else:
            raise StopIteration


cdef class Float64toFloat64MapView:
    cdef Float64toFloat64MapIterator get_iter(self):
        return Float64toFloat64MapIterator(self.parent, self.view_type)  

    def __cinit__(self, Float64toFloat64Map parent, view_type):
        self.parent = parent
        self.view_type = view_type

    def __iter__(self):
        return self.get_iter()

    def __len__(self):
        return self.parent.size()

    def __contains__(self, x):
        for y in self:
            if x==y:
                return True
        return False

##########################      Utils:
cpdef Float64toFloat64Map Float64toFloat64Map_from_buffers(float64_t[:] keys, float64_t[:] vals, double size_hint=0.0):
    cdef Py_ssize_t n = len(keys)
    cdef Py_ssize_t b = len(vals)
    if b < n:
        n = b
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Float64toFloat64Map(number_of_elements_hint=at_least_needed)
    cdef Py_ssize_t i
    for i in range(n):
        res.cput(keys[i], vals[i])
    return res

cdef float64_t DEFAULT_VALUE_float64tofloat64 = float("nan")
cpdef size_t Float64toFloat64Map_to(Float64toFloat64Map map, float64_t[:] keys, float64_t[:] vals, bint stop_at_unknown=True, float64_t default_value=DEFAULT_VALUE_float64tofloat64) except *:
    """returns number of found keys"""
    if map is None:
        raise TypeError("'NoneType' is not a map")
    cdef size_t n = len(keys)
    if n != len(vals):
        raise ValueError("Different lengths of keys and vals arrays")
    cdef size_t i
    cdef khint_t k
    cdef size_t res = 0
    for i in range(n):
        k = kh_get_float64tofloat64map(map.table, keys[i])
        if k != map.table.n_buckets:
            vals[i] = map.table.vals[k]
            res += 1
        else:
            vals[i] = default_value
            if stop_at_unknown:
                return res
    return res

cpdef void swap_float64tofloat64map(Float64toFloat64Map a, Float64toFloat64Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef kh_float64tofloat64map_t *tmp=a.table
    a.table=b.table
    b.table=tmp


cpdef Float64toFloat64Map copy_float64tofloat64map(Float64toFloat64Map s):
    if s is None:
        return None
    cdef Float64toFloat64Map result = Float64toFloat64Map(number_of_elements_hint=s.size())
    cdef Float64toFloat64MapIterator it=s.get_iter(2)
    cdef float64tofloat64_key_val_pair p
    while it.has_next():
        p = it.next()
        result.cput(p.key, p.val)
    return result


cpdef bint are_equal_float64tofloat64map(Float64toFloat64Map a, Float64toFloat64Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    if a.size()!=b.size():
        return False
    cdef Float64toFloat64MapIterator it=a.get_iter(2)
    cdef float64tofloat64_key_val_pair p
    while it.has_next():
        p = it.next()
        if not b.contains(p.key):
            return False
    return True


cpdef void update_float64tofloat64map(Float64toFloat64Map a, Float64toFloat64Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    cdef Float64toFloat64MapIterator it=b.get_iter(2)
    cdef float64tofloat64_key_val_pair p
    while it.has_next():
        p = it.next()
        a.cput(p.key, p.val)


cdef class Int32toInt32Map:

    @classmethod
    def fromkeys(cls, iterable, value):
        return Int32toInt32Map(((key, value) for key in iterable))

    def __cinit__(self, iterable=None, *, number_of_elements_hint=None):
        """
        number_of_elements_hint - number of elements without the need of reallocation.
        """
        self.table = kh_init_int32toint32map()
        if number_of_elements_hint is not None:
            kh_resize_int32toint32map(self.table, element_n_to_bucket_n(number_of_elements_hint))
        cdef int32_t key
        cdef int32_t val
        if iterable is not None:
            for key, val in iterable:
                    self.cput(key, val)

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_int32toint32map(self.table)
            self.table = NULL

    cpdef void discard(self, int32_t key) except *:
        cdef khint_t k
        k = kh_get_int32toint32map(self.table, key)
        if k != self.table.n_buckets:
            kh_del_int32toint32map(self.table, k)

    cdef bint contains(self, int32_t key) except *:
        cdef khint_t k
        k = kh_get_int32toint32map(self.table, key)
        return k != self.table.n_buckets

    def __contains__(self, int32_t key):
        return self.contains(key)


    def __len__(self):
        return self.size()
  
    cdef khint_t size(self):
        return self.table.size

    cpdef void cput(self, int32_t key, int32_t val) except *:
        cdef:
            khint_t k
            int ret = 0

        k = kh_put_int32toint32map(self.table, key, &ret)
        self.table.keys[k] = key
        self.table.vals[k] = val

    def __setitem__(self, key, val):
        self.cput(key, val)

    cpdef int32_t cget(self, int32_t key) except *:
        k = kh_get_int32toint32map(self.table, key)
        if k != self.table.n_buckets:
            return self.table.vals[k]
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        return self.cget(key)
        


    cdef Int32toInt32MapIterator get_iter(self, int view_type):
        return Int32toInt32MapIterator(self, view_type)

    def clear(self):
        cdef Int32toInt32Map tmp=Int32toInt32Map()
        swap_int32toint32map(self, tmp)

    def copy(self):
        return copy_int32toint32map(self)

    def update(self, other):
        if isinstance(other, Int32toInt32Map):
            update_int32toint32map(self, other)
            return
        for key,val in other:
            self[key]=val

    def setdefault(self, key, default):
        try:
            return self[key]
        except KeyError:
            self[key]=default
            return default

    def get(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("get() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("get() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("get() takes no keyword arguments")
        key = args[0]
        try:
            return self[key]
        except KeyError:
            if len(args)==1:
                return None
            return args[1]

    def pop(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("pop() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("pop() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("pop() takes no keyword arguments")
        key = args[0]
        try:
            val = self[key]
        except KeyError as e:
            if len(args)==1:
                raise e from None
            return args[1]
        del self[key]
        return val

    def popitem(self):
        if self.size()== 0:
            raise KeyError("popitem(): dictionary is empty")
        key = next(iter(self))
        val = self.pop(key)
        return (key, val)

    def keys(self):
        return Int32toInt32MapView(self, 0)

    def values(self):
        return Int32toInt32MapView(self, 1)

    def items(self):
        return Int32toInt32MapView(self, 2)

    def __iter__(self):
        return iter(self.keys())

    def __delitem__(self, key):
        cdef size_t old=self.size()
        self.discard(key)
        if old==self.size():
            raise KeyError(key)

    def __eq__(self, other):
        return are_equal_int32toint32map(self,other)


### Iterator:
cdef class Int32toInt32MapIterator:

    cdef void __move(self) except *:
        while self.it<self.parent.table.n_buckets and not kh_exist_int32toint32map(self.parent.table, self.it):
              self.it+=1       

    cdef bint has_next(self) except *:
        self.__move()
        return self.it < self.parent.table.n_buckets
      
    # doesn't work if there was change between last has_next() and next()       
    cdef int32toint32_key_val_pair next(self) except *:
        cdef int32toint32_key_val_pair result 
        result.key = self.parent.table.keys[self.it]
        result.val = self.parent.table.vals[self.it]
        self.it+=1#ensure at least one move!
        return result

    def __cinit__(self, Int32toInt32Map parent, view_type):
        self.parent = parent
        self.view_type = view_type
        #search the start:
        self.it = 0
        self.__move()

    def __next__(self):
        cdef int32toint32_key_val_pair pair
        if self.has_next():
            pair=self.next()

            if self.view_type == 0:           # keys
                return pair.key
            if self.view_type == 1:           # vals
                return pair.val
            else:                        # items
                return (pair.key, pair.val)
        else:
            raise StopIteration


cdef class Int32toInt32MapView:
    cdef Int32toInt32MapIterator get_iter(self):
        return Int32toInt32MapIterator(self.parent, self.view_type)  

    def __cinit__(self, Int32toInt32Map parent, view_type):
        self.parent = parent
        self.view_type = view_type

    def __iter__(self):
        return self.get_iter()

    def __len__(self):
        return self.parent.size()

    def __contains__(self, x):
        for y in self:
            if x==y:
                return True
        return False

##########################      Utils:
cpdef Int32toInt32Map Int32toInt32Map_from_buffers(int32_t[:] keys, int32_t[:] vals, double size_hint=0.0):
    cdef Py_ssize_t n = len(keys)
    cdef Py_ssize_t b = len(vals)
    if b < n:
        n = b
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Int32toInt32Map(number_of_elements_hint=at_least_needed)
    cdef Py_ssize_t i
    for i in range(n):
        res.cput(keys[i], vals[i])
    return res

cdef int32_t DEFAULT_VALUE_int32toint32 = 0
cpdef size_t Int32toInt32Map_to(Int32toInt32Map map, int32_t[:] keys, int32_t[:] vals, bint stop_at_unknown=True, int32_t default_value=DEFAULT_VALUE_int32toint32) except *:
    """returns number of found keys"""
    if map is None:
        raise TypeError("'NoneType' is not a map")
    cdef size_t n = len(keys)
    if n != len(vals):
        raise ValueError("Different lengths of keys and vals arrays")
    cdef size_t i
    cdef khint_t k
    cdef size_t res = 0
    for i in range(n):
        k = kh_get_int32toint32map(map.table, keys[i])
        if k != map.table.n_buckets:
            vals[i] = map.table.vals[k]
            res += 1
        else:
            vals[i] = default_value
            if stop_at_unknown:
                return res
    return res

cpdef void swap_int32toint32map(Int32toInt32Map a, Int32toInt32Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef kh_int32toint32map_t *tmp=a.table
    a.table=b.table
    b.table=tmp


cpdef Int32toInt32Map copy_int32toint32map(Int32toInt32Map s):
    if s is None:
        return None
    cdef Int32toInt32Map result = Int32toInt32Map(number_of_elements_hint=s.size())
    cdef Int32toInt32MapIterator it=s.get_iter(2)
    cdef int32toint32_key_val_pair p
    while it.has_next():
        p = it.next()
        result.cput(p.key, p.val)
    return result


cpdef bint are_equal_int32toint32map(Int32toInt32Map a, Int32toInt32Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    if a.size()!=b.size():
        return False
    cdef Int32toInt32MapIterator it=a.get_iter(2)
    cdef int32toint32_key_val_pair p
    while it.has_next():
        p = it.next()
        if not b.contains(p.key):
            return False
    return True


cpdef void update_int32toint32map(Int32toInt32Map a, Int32toInt32Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    cdef Int32toInt32MapIterator it=b.get_iter(2)
    cdef int32toint32_key_val_pair p
    while it.has_next():
        p = it.next()
        a.cput(p.key, p.val)


cdef class Int32toFloat32Map:

    @classmethod
    def fromkeys(cls, iterable, value):
        return Int32toFloat32Map(((key, value) for key in iterable))

    def __cinit__(self, iterable=None, *, number_of_elements_hint=None):
        """
        number_of_elements_hint - number of elements without the need of reallocation.
        """
        self.table = kh_init_int32tofloat32map()
        if number_of_elements_hint is not None:
            kh_resize_int32tofloat32map(self.table, element_n_to_bucket_n(number_of_elements_hint))
        cdef int32_t key
        cdef float32_t val
        if iterable is not None:
            for key, val in iterable:
                    self.cput(key, val)

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_int32tofloat32map(self.table)
            self.table = NULL

    cpdef void discard(self, int32_t key) except *:
        cdef khint_t k
        k = kh_get_int32tofloat32map(self.table, key)
        if k != self.table.n_buckets:
            kh_del_int32tofloat32map(self.table, k)

    cdef bint contains(self, int32_t key) except *:
        cdef khint_t k
        k = kh_get_int32tofloat32map(self.table, key)
        return k != self.table.n_buckets

    def __contains__(self, int32_t key):
        return self.contains(key)


    def __len__(self):
        return self.size()
  
    cdef khint_t size(self):
        return self.table.size

    cpdef void cput(self, int32_t key, float32_t val) except *:
        cdef:
            khint_t k
            int ret = 0

        k = kh_put_int32tofloat32map(self.table, key, &ret)
        self.table.keys[k] = key
        self.table.vals[k] = val

    def __setitem__(self, key, val):
        self.cput(key, val)

    cpdef float32_t cget(self, int32_t key) except *:
        k = kh_get_int32tofloat32map(self.table, key)
        if k != self.table.n_buckets:
            return self.table.vals[k]
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        return self.cget(key)
        


    cdef Int32toFloat32MapIterator get_iter(self, int view_type):
        return Int32toFloat32MapIterator(self, view_type)

    def clear(self):
        cdef Int32toFloat32Map tmp=Int32toFloat32Map()
        swap_int32tofloat32map(self, tmp)

    def copy(self):
        return copy_int32tofloat32map(self)

    def update(self, other):
        if isinstance(other, Int32toFloat32Map):
            update_int32tofloat32map(self, other)
            return
        for key,val in other:
            self[key]=val

    def setdefault(self, key, default):
        try:
            return self[key]
        except KeyError:
            self[key]=default
            return default

    def get(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("get() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("get() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("get() takes no keyword arguments")
        key = args[0]
        try:
            return self[key]
        except KeyError:
            if len(args)==1:
                return None
            return args[1]

    def pop(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("pop() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("pop() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("pop() takes no keyword arguments")
        key = args[0]
        try:
            val = self[key]
        except KeyError as e:
            if len(args)==1:
                raise e from None
            return args[1]
        del self[key]
        return val

    def popitem(self):
        if self.size()== 0:
            raise KeyError("popitem(): dictionary is empty")
        key = next(iter(self))
        val = self.pop(key)
        return (key, val)

    def keys(self):
        return Int32toFloat32MapView(self, 0)

    def values(self):
        return Int32toFloat32MapView(self, 1)

    def items(self):
        return Int32toFloat32MapView(self, 2)

    def __iter__(self):
        return iter(self.keys())

    def __delitem__(self, key):
        cdef size_t old=self.size()
        self.discard(key)
        if old==self.size():
            raise KeyError(key)

    def __eq__(self, other):
        return are_equal_int32tofloat32map(self,other)


### Iterator:
cdef class Int32toFloat32MapIterator:

    cdef void __move(self) except *:
        while self.it<self.parent.table.n_buckets and not kh_exist_int32tofloat32map(self.parent.table, self.it):
              self.it+=1       

    cdef bint has_next(self) except *:
        self.__move()
        return self.it < self.parent.table.n_buckets
      
    # doesn't work if there was change between last has_next() and next()       
    cdef int32tofloat32_key_val_pair next(self) except *:
        cdef int32tofloat32_key_val_pair result 
        result.key = self.parent.table.keys[self.it]
        result.val = self.parent.table.vals[self.it]
        self.it+=1#ensure at least one move!
        return result

    def __cinit__(self, Int32toFloat32Map parent, view_type):
        self.parent = parent
        self.view_type = view_type
        #search the start:
        self.it = 0
        self.__move()

    def __next__(self):
        cdef int32tofloat32_key_val_pair pair
        if self.has_next():
            pair=self.next()

            if self.view_type == 0:           # keys
                return pair.key
            if self.view_type == 1:           # vals
                return pair.val
            else:                        # items
                return (pair.key, pair.val)
        else:
            raise StopIteration


cdef class Int32toFloat32MapView:
    cdef Int32toFloat32MapIterator get_iter(self):
        return Int32toFloat32MapIterator(self.parent, self.view_type)  

    def __cinit__(self, Int32toFloat32Map parent, view_type):
        self.parent = parent
        self.view_type = view_type

    def __iter__(self):
        return self.get_iter()

    def __len__(self):
        return self.parent.size()

    def __contains__(self, x):
        for y in self:
            if x==y:
                return True
        return False

##########################      Utils:
cpdef Int32toFloat32Map Int32toFloat32Map_from_buffers(int32_t[:] keys, float32_t[:] vals, double size_hint=0.0):
    cdef Py_ssize_t n = len(keys)
    cdef Py_ssize_t b = len(vals)
    if b < n:
        n = b
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Int32toFloat32Map(number_of_elements_hint=at_least_needed)
    cdef Py_ssize_t i
    for i in range(n):
        res.cput(keys[i], vals[i])
    return res

cdef float32_t DEFAULT_VALUE_int32tofloat32 = float("nan")
cpdef size_t Int32toFloat32Map_to(Int32toFloat32Map map, int32_t[:] keys, float32_t[:] vals, bint stop_at_unknown=True, float32_t default_value=DEFAULT_VALUE_int32tofloat32) except *:
    """returns number of found keys"""
    if map is None:
        raise TypeError("'NoneType' is not a map")
    cdef size_t n = len(keys)
    if n != len(vals):
        raise ValueError("Different lengths of keys and vals arrays")
    cdef size_t i
    cdef khint_t k
    cdef size_t res = 0
    for i in range(n):
        k = kh_get_int32tofloat32map(map.table, keys[i])
        if k != map.table.n_buckets:
            vals[i] = map.table.vals[k]
            res += 1
        else:
            vals[i] = default_value
            if stop_at_unknown:
                return res
    return res

cpdef void swap_int32tofloat32map(Int32toFloat32Map a, Int32toFloat32Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef kh_int32tofloat32map_t *tmp=a.table
    a.table=b.table
    b.table=tmp


cpdef Int32toFloat32Map copy_int32tofloat32map(Int32toFloat32Map s):
    if s is None:
        return None
    cdef Int32toFloat32Map result = Int32toFloat32Map(number_of_elements_hint=s.size())
    cdef Int32toFloat32MapIterator it=s.get_iter(2)
    cdef int32tofloat32_key_val_pair p
    while it.has_next():
        p = it.next()
        result.cput(p.key, p.val)
    return result


cpdef bint are_equal_int32tofloat32map(Int32toFloat32Map a, Int32toFloat32Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    if a.size()!=b.size():
        return False
    cdef Int32toFloat32MapIterator it=a.get_iter(2)
    cdef int32tofloat32_key_val_pair p
    while it.has_next():
        p = it.next()
        if not b.contains(p.key):
            return False
    return True


cpdef void update_int32tofloat32map(Int32toFloat32Map a, Int32toFloat32Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    cdef Int32toFloat32MapIterator it=b.get_iter(2)
    cdef int32tofloat32_key_val_pair p
    while it.has_next():
        p = it.next()
        a.cput(p.key, p.val)


cdef class Float32toInt32Map:

    @classmethod
    def fromkeys(cls, iterable, value):
        return Float32toInt32Map(((key, value) for key in iterable))

    def __cinit__(self, iterable=None, *, number_of_elements_hint=None):
        """
        number_of_elements_hint - number of elements without the need of reallocation.
        """
        self.table = kh_init_float32toint32map()
        if number_of_elements_hint is not None:
            kh_resize_float32toint32map(self.table, element_n_to_bucket_n(number_of_elements_hint))
        cdef float32_t key
        cdef int32_t val
        if iterable is not None:
            for key, val in iterable:
                    self.cput(key, val)

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_float32toint32map(self.table)
            self.table = NULL

    cpdef void discard(self, float32_t key) except *:
        cdef khint_t k
        k = kh_get_float32toint32map(self.table, key)
        if k != self.table.n_buckets:
            kh_del_float32toint32map(self.table, k)

    cdef bint contains(self, float32_t key) except *:
        cdef khint_t k
        k = kh_get_float32toint32map(self.table, key)
        return k != self.table.n_buckets

    def __contains__(self, float32_t key):
        return self.contains(key)


    def __len__(self):
        return self.size()
  
    cdef khint_t size(self):
        return self.table.size

    cpdef void cput(self, float32_t key, int32_t val) except *:
        cdef:
            khint_t k
            int ret = 0

        k = kh_put_float32toint32map(self.table, key, &ret)
        self.table.keys[k] = key
        self.table.vals[k] = val

    def __setitem__(self, key, val):
        self.cput(key, val)

    cpdef int32_t cget(self, float32_t key) except *:
        k = kh_get_float32toint32map(self.table, key)
        if k != self.table.n_buckets:
            return self.table.vals[k]
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        return self.cget(key)
        


    cdef Float32toInt32MapIterator get_iter(self, int view_type):
        return Float32toInt32MapIterator(self, view_type)

    def clear(self):
        cdef Float32toInt32Map tmp=Float32toInt32Map()
        swap_float32toint32map(self, tmp)

    def copy(self):
        return copy_float32toint32map(self)

    def update(self, other):
        if isinstance(other, Float32toInt32Map):
            update_float32toint32map(self, other)
            return
        for key,val in other:
            self[key]=val

    def setdefault(self, key, default):
        try:
            return self[key]
        except KeyError:
            self[key]=default
            return default

    def get(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("get() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("get() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("get() takes no keyword arguments")
        key = args[0]
        try:
            return self[key]
        except KeyError:
            if len(args)==1:
                return None
            return args[1]

    def pop(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("pop() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("pop() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("pop() takes no keyword arguments")
        key = args[0]
        try:
            val = self[key]
        except KeyError as e:
            if len(args)==1:
                raise e from None
            return args[1]
        del self[key]
        return val

    def popitem(self):
        if self.size()== 0:
            raise KeyError("popitem(): dictionary is empty")
        key = next(iter(self))
        val = self.pop(key)
        return (key, val)

    def keys(self):
        return Float32toInt32MapView(self, 0)

    def values(self):
        return Float32toInt32MapView(self, 1)

    def items(self):
        return Float32toInt32MapView(self, 2)

    def __iter__(self):
        return iter(self.keys())

    def __delitem__(self, key):
        cdef size_t old=self.size()
        self.discard(key)
        if old==self.size():
            raise KeyError(key)

    def __eq__(self, other):
        return are_equal_float32toint32map(self,other)


### Iterator:
cdef class Float32toInt32MapIterator:

    cdef void __move(self) except *:
        while self.it<self.parent.table.n_buckets and not kh_exist_float32toint32map(self.parent.table, self.it):
              self.it+=1       

    cdef bint has_next(self) except *:
        self.__move()
        return self.it < self.parent.table.n_buckets
      
    # doesn't work if there was change between last has_next() and next()       
    cdef float32toint32_key_val_pair next(self) except *:
        cdef float32toint32_key_val_pair result 
        result.key = self.parent.table.keys[self.it]
        result.val = self.parent.table.vals[self.it]
        self.it+=1#ensure at least one move!
        return result

    def __cinit__(self, Float32toInt32Map parent, view_type):
        self.parent = parent
        self.view_type = view_type
        #search the start:
        self.it = 0
        self.__move()

    def __next__(self):
        cdef float32toint32_key_val_pair pair
        if self.has_next():
            pair=self.next()

            if self.view_type == 0:           # keys
                return pair.key
            if self.view_type == 1:           # vals
                return pair.val
            else:                        # items
                return (pair.key, pair.val)
        else:
            raise StopIteration


cdef class Float32toInt32MapView:
    cdef Float32toInt32MapIterator get_iter(self):
        return Float32toInt32MapIterator(self.parent, self.view_type)  

    def __cinit__(self, Float32toInt32Map parent, view_type):
        self.parent = parent
        self.view_type = view_type

    def __iter__(self):
        return self.get_iter()

    def __len__(self):
        return self.parent.size()

    def __contains__(self, x):
        for y in self:
            if x==y:
                return True
        return False

##########################      Utils:
cpdef Float32toInt32Map Float32toInt32Map_from_buffers(float32_t[:] keys, int32_t[:] vals, double size_hint=0.0):
    cdef Py_ssize_t n = len(keys)
    cdef Py_ssize_t b = len(vals)
    if b < n:
        n = b
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Float32toInt32Map(number_of_elements_hint=at_least_needed)
    cdef Py_ssize_t i
    for i in range(n):
        res.cput(keys[i], vals[i])
    return res

cdef int32_t DEFAULT_VALUE_float32toint32 = 0
cpdef size_t Float32toInt32Map_to(Float32toInt32Map map, float32_t[:] keys, int32_t[:] vals, bint stop_at_unknown=True, int32_t default_value=DEFAULT_VALUE_float32toint32) except *:
    """returns number of found keys"""
    if map is None:
        raise TypeError("'NoneType' is not a map")
    cdef size_t n = len(keys)
    if n != len(vals):
        raise ValueError("Different lengths of keys and vals arrays")
    cdef size_t i
    cdef khint_t k
    cdef size_t res = 0
    for i in range(n):
        k = kh_get_float32toint32map(map.table, keys[i])
        if k != map.table.n_buckets:
            vals[i] = map.table.vals[k]
            res += 1
        else:
            vals[i] = default_value
            if stop_at_unknown:
                return res
    return res

cpdef void swap_float32toint32map(Float32toInt32Map a, Float32toInt32Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef kh_float32toint32map_t *tmp=a.table
    a.table=b.table
    b.table=tmp


cpdef Float32toInt32Map copy_float32toint32map(Float32toInt32Map s):
    if s is None:
        return None
    cdef Float32toInt32Map result = Float32toInt32Map(number_of_elements_hint=s.size())
    cdef Float32toInt32MapIterator it=s.get_iter(2)
    cdef float32toint32_key_val_pair p
    while it.has_next():
        p = it.next()
        result.cput(p.key, p.val)
    return result


cpdef bint are_equal_float32toint32map(Float32toInt32Map a, Float32toInt32Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    if a.size()!=b.size():
        return False
    cdef Float32toInt32MapIterator it=a.get_iter(2)
    cdef float32toint32_key_val_pair p
    while it.has_next():
        p = it.next()
        if not b.contains(p.key):
            return False
    return True


cpdef void update_float32toint32map(Float32toInt32Map a, Float32toInt32Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    cdef Float32toInt32MapIterator it=b.get_iter(2)
    cdef float32toint32_key_val_pair p
    while it.has_next():
        p = it.next()
        a.cput(p.key, p.val)


cdef class Float32toFloat32Map:

    @classmethod
    def fromkeys(cls, iterable, value):
        return Float32toFloat32Map(((key, value) for key in iterable))

    def __cinit__(self, iterable=None, *, number_of_elements_hint=None):
        """
        number_of_elements_hint - number of elements without the need of reallocation.
        """
        self.table = kh_init_float32tofloat32map()
        if number_of_elements_hint is not None:
            kh_resize_float32tofloat32map(self.table, element_n_to_bucket_n(number_of_elements_hint))
        cdef float32_t key
        cdef float32_t val
        if iterable is not None:
            for key, val in iterable:
                    self.cput(key, val)

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_float32tofloat32map(self.table)
            self.table = NULL

    cpdef void discard(self, float32_t key) except *:
        cdef khint_t k
        k = kh_get_float32tofloat32map(self.table, key)
        if k != self.table.n_buckets:
            kh_del_float32tofloat32map(self.table, k)

    cdef bint contains(self, float32_t key) except *:
        cdef khint_t k
        k = kh_get_float32tofloat32map(self.table, key)
        return k != self.table.n_buckets

    def __contains__(self, float32_t key):
        return self.contains(key)


    def __len__(self):
        return self.size()
  
    cdef khint_t size(self):
        return self.table.size

    cpdef void cput(self, float32_t key, float32_t val) except *:
        cdef:
            khint_t k
            int ret = 0

        k = kh_put_float32tofloat32map(self.table, key, &ret)
        self.table.keys[k] = key
        self.table.vals[k] = val

    def __setitem__(self, key, val):
        self.cput(key, val)

    cpdef float32_t cget(self, float32_t key) except *:
        k = kh_get_float32tofloat32map(self.table, key)
        if k != self.table.n_buckets:
            return self.table.vals[k]
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        return self.cget(key)
        


    cdef Float32toFloat32MapIterator get_iter(self, int view_type):
        return Float32toFloat32MapIterator(self, view_type)

    def clear(self):
        cdef Float32toFloat32Map tmp=Float32toFloat32Map()
        swap_float32tofloat32map(self, tmp)

    def copy(self):
        return copy_float32tofloat32map(self)

    def update(self, other):
        if isinstance(other, Float32toFloat32Map):
            update_float32tofloat32map(self, other)
            return
        for key,val in other:
            self[key]=val

    def setdefault(self, key, default):
        try:
            return self[key]
        except KeyError:
            self[key]=default
            return default

    def get(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("get() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("get() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("get() takes no keyword arguments")
        key = args[0]
        try:
            return self[key]
        except KeyError:
            if len(args)==1:
                return None
            return args[1]

    def pop(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("pop() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("pop() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("pop() takes no keyword arguments")
        key = args[0]
        try:
            val = self[key]
        except KeyError as e:
            if len(args)==1:
                raise e from None
            return args[1]
        del self[key]
        return val

    def popitem(self):
        if self.size()== 0:
            raise KeyError("popitem(): dictionary is empty")
        key = next(iter(self))
        val = self.pop(key)
        return (key, val)

    def keys(self):
        return Float32toFloat32MapView(self, 0)

    def values(self):
        return Float32toFloat32MapView(self, 1)

    def items(self):
        return Float32toFloat32MapView(self, 2)

    def __iter__(self):
        return iter(self.keys())

    def __delitem__(self, key):
        cdef size_t old=self.size()
        self.discard(key)
        if old==self.size():
            raise KeyError(key)

    def __eq__(self, other):
        return are_equal_float32tofloat32map(self,other)


### Iterator:
cdef class Float32toFloat32MapIterator:

    cdef void __move(self) except *:
        while self.it<self.parent.table.n_buckets and not kh_exist_float32tofloat32map(self.parent.table, self.it):
              self.it+=1       

    cdef bint has_next(self) except *:
        self.__move()
        return self.it < self.parent.table.n_buckets
      
    # doesn't work if there was change between last has_next() and next()       
    cdef float32tofloat32_key_val_pair next(self) except *:
        cdef float32tofloat32_key_val_pair result 
        result.key = self.parent.table.keys[self.it]
        result.val = self.parent.table.vals[self.it]
        self.it+=1#ensure at least one move!
        return result

    def __cinit__(self, Float32toFloat32Map parent, view_type):
        self.parent = parent
        self.view_type = view_type
        #search the start:
        self.it = 0
        self.__move()

    def __next__(self):
        cdef float32tofloat32_key_val_pair pair
        if self.has_next():
            pair=self.next()

            if self.view_type == 0:           # keys
                return pair.key
            if self.view_type == 1:           # vals
                return pair.val
            else:                        # items
                return (pair.key, pair.val)
        else:
            raise StopIteration


cdef class Float32toFloat32MapView:
    cdef Float32toFloat32MapIterator get_iter(self):
        return Float32toFloat32MapIterator(self.parent, self.view_type)  

    def __cinit__(self, Float32toFloat32Map parent, view_type):
        self.parent = parent
        self.view_type = view_type

    def __iter__(self):
        return self.get_iter()

    def __len__(self):
        return self.parent.size()

    def __contains__(self, x):
        for y in self:
            if x==y:
                return True
        return False

##########################      Utils:
cpdef Float32toFloat32Map Float32toFloat32Map_from_buffers(float32_t[:] keys, float32_t[:] vals, double size_hint=0.0):
    cdef Py_ssize_t n = len(keys)
    cdef Py_ssize_t b = len(vals)
    if b < n:
        n = b
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Float32toFloat32Map(number_of_elements_hint=at_least_needed)
    cdef Py_ssize_t i
    for i in range(n):
        res.cput(keys[i], vals[i])
    return res

cdef float32_t DEFAULT_VALUE_float32tofloat32 = float("nan")
cpdef size_t Float32toFloat32Map_to(Float32toFloat32Map map, float32_t[:] keys, float32_t[:] vals, bint stop_at_unknown=True, float32_t default_value=DEFAULT_VALUE_float32tofloat32) except *:
    """returns number of found keys"""
    if map is None:
        raise TypeError("'NoneType' is not a map")
    cdef size_t n = len(keys)
    if n != len(vals):
        raise ValueError("Different lengths of keys and vals arrays")
    cdef size_t i
    cdef khint_t k
    cdef size_t res = 0
    for i in range(n):
        k = kh_get_float32tofloat32map(map.table, keys[i])
        if k != map.table.n_buckets:
            vals[i] = map.table.vals[k]
            res += 1
        else:
            vals[i] = default_value
            if stop_at_unknown:
                return res
    return res

cpdef void swap_float32tofloat32map(Float32toFloat32Map a, Float32toFloat32Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef kh_float32tofloat32map_t *tmp=a.table
    a.table=b.table
    b.table=tmp


cpdef Float32toFloat32Map copy_float32tofloat32map(Float32toFloat32Map s):
    if s is None:
        return None
    cdef Float32toFloat32Map result = Float32toFloat32Map(number_of_elements_hint=s.size())
    cdef Float32toFloat32MapIterator it=s.get_iter(2)
    cdef float32tofloat32_key_val_pair p
    while it.has_next():
        p = it.next()
        result.cput(p.key, p.val)
    return result


cpdef bint are_equal_float32tofloat32map(Float32toFloat32Map a, Float32toFloat32Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    if a.size()!=b.size():
        return False
    cdef Float32toFloat32MapIterator it=a.get_iter(2)
    cdef float32tofloat32_key_val_pair p
    while it.has_next():
        p = it.next()
        if not b.contains(p.key):
            return False
    return True


cpdef void update_float32tofloat32map(Float32toFloat32Map a, Float32toFloat32Map b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    cdef Float32toFloat32MapIterator it=b.get_iter(2)
    cdef float32tofloat32_key_val_pair p
    while it.has_next():
        p = it.next()
        a.cput(p.key, p.val)


cdef class PyObjectMap:

    @classmethod
    def fromkeys(cls, iterable, value):
        return PyObjectMap((key, value) for key in iterable)

    def __cinit__(self, iterable=None, *, number_of_elements_hint=None):
        """
        number_of_elements_hint - number of elements without the need of reallocation.
        """
        self.table = kh_init_pyobjectmap()
        if number_of_elements_hint is not None:
            kh_resize_pyobjectmap(self.table, element_n_to_bucket_n(number_of_elements_hint))
        if iterable is not None:
            for key, val in iterable:
                    self.cput(key, val)

 
    cpdef void discard(self, object key) except *:
        cdef khint_t k
        k = kh_get_pyobjectmap(self.table, <pyobject_t>key)
        if k != self.table.n_buckets:
            Py_DECREF(<object>(self.table.keys[k]))
            Py_DECREF(<object>(self.table.vals[k]))
            kh_del_pyobjectmap(self.table, k)

    def __dealloc__(self):
        cdef Py_ssize_t i
        if self.table is not NULL:
            for i in range(self.table.size):
                if kh_exist_pyobjectmap(self.table, i):
                    Py_DECREF(<object>(self.table.keys[i]))
                    Py_DECREF(<object>(self.table.vals[i]))
            kh_destroy_pyobjectmap(self.table)
            self.table = NULL

    cdef bint contains(self, pyobject_t key) except *:
        cdef khint_t k
        k = kh_get_pyobjectmap(self.table, key)
        return k != self.table.n_buckets

    def __contains__(self, object key):
        return self.contains(<pyobject_t>key)


    def __len__(self):
        return self.size()
  
    cdef khint_t size(self):
        return self.table.size

    cpdef void cput(self, object key, object val) except *:
        cdef:
            khint_t k
            int ret = 0
        k = kh_put_pyobjectmap(self.table, <pyobject_t>key, &ret)
        if not ret:
            Py_DECREF(<object>(self.table.vals[k]))
        else:
            Py_INCREF(key)
        Py_INCREF(val)
        self.table.vals[k] = <pyobject_t> val
 
    def __setitem__(self, key, val):
        self.cput(key, val)

    cpdef object cget(self, object key):
        k = kh_get_pyobjectmap(self.table, <pyobject_t>key)
        if k != self.table.n_buckets:
            return <object>self.table.vals[k]
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        return self.cget(key)

    cdef PyObjectMapIterator get_iter(self, int view_type):
        return PyObjectMapIterator(self, view_type)

    def clear(self):
        cdef PyObjectMap tmp=PyObjectMap()
        swap_pyobjectmap(self, tmp)

    def copy(self):
        return copy_pyobjectmap(self)

    def update(self, other):
        if isinstance(other, PyObjectMap):
            update_pyobjectmap(self, other)
            return
        for key,val in other:
            self[key]=val

    def setdefault(self, key, default):
        try:
            return self[key]
        except KeyError:
            self[key]=default
            return default

    def get(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("get() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("get() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("get() takes no keyword arguments")
        key = args[0]
        try:
            return self[key]
        except KeyError:
            if len(args)==1:
                return None
            return args[1]

    def pop(self, *args, **kwargs):
        if len(args)==0:
            raise TypeError("pop() expected at least 1 arguments, got 0")
        if len(args)>2:
            raise TypeError("pop() expected at most 2 arguments, got {0}".format(len(args)))
        if kwargs:
            raise TypeError("pop() takes no keyword arguments")
        key = args[0]
        try:
            val = self[key]
        except KeyError as e:
            if len(args)==1:
                raise e from None
            return args[1]
        del self[key]
        return val

    def popitem(self):
        if self.size()== 0:
            raise KeyError("popitem(): dictionary is empty")
        key = next(iter(self))
        val = self.pop(key)
        return (key, val)

    def keys(self):
        return PyObjectMapView(self, 0)

    def values(self):
        return PyObjectMapView(self, 1)

    def items(self):
        return PyObjectMapView(self, 2)

    def __iter__(self):
        return iter(self.keys())

    def __delitem__(self, key):
        cdef size_t old=self.size()
        self.discard(key)
        if old==self.size():
            raise KeyError(key)

    def __eq__(self, other):
        return are_equal_pyobjectmap(self,other)


### Iterator:
cdef class PyObjectMapIterator:

    cdef void __move(self) except *:
        while self.it<self.parent.table.n_buckets and not kh_exist_pyobjectmap(self.parent.table, self.it):
              self.it+=1       

    cdef bint has_next(self) except *:
        self.__move()
        return self.it < self.parent.table.n_buckets
      
    # doesn't work if there was change between last has_next() and next()       
    cdef pyobject_key_val_pair next(self) except *:
        cdef pyobject_key_val_pair result 
        result.key = self.parent.table.keys[self.it]
        result.val = self.parent.table.vals[self.it]
        self.it+=1#ensure at least one move!
        return result

    def __cinit__(self, PyObjectMap parent, view_type):
        self.parent = parent
        self.view_type = view_type
        #search the start:
        self.it = 0
        self.__move()

    def __next__(self):
        cdef pyobject_key_val_pair pair
        if self.has_next():
            pair=self.next()

            if self.view_type == 0:           # keys
                return <object>pair.key
            if self.view_type == 1:           # vals
                return <object>pair.val
            else:                            # items
                return (<object>pair.key, <object>pair.val)

        else:
            raise StopIteration


cdef class PyObjectMapView:
    cdef PyObjectMapIterator get_iter(self):
        return PyObjectMapIterator(self.parent, self.view_type)  

    def __cinit__(self, PyObjectMap parent, view_type):
        self.parent = parent
        self.view_type = view_type

    def __iter__(self):
        return self.get_iter()

    def __len__(self):
        return self.parent.size()

    def __contains__(self, x):
        for y in self:
            if x==y:
                return True
        return False

##########################      Utils:
cpdef PyObjectMap PyObjectMap_from_buffers(object[:] keys, object[:] vals, double size_hint=0.0):
    cdef Py_ssize_t n = len(keys)
    cdef Py_ssize_t b = len(vals)
    if b < n:
        n = b
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=PyObjectMap(number_of_elements_hint=at_least_needed)
    cdef Py_ssize_t i
    for i in range(n):
        res.cput(keys[i], vals[i])
    return res

cdef object DEFAULT_VALUE_pyobject = None
cpdef size_t PyObjectMap_to(PyObjectMap map, object[:] keys, object[:] vals, bint stop_at_unknown=True, object default_value=None) except *:
    """returns number of found keys"""
    if map is None:
        raise TypeError("'NoneType' is not a map")
    cdef size_t n = len(keys)
    if n != len(vals):
        raise ValueError("Different lengths of keys and vals arrays")
    cdef size_t i
    cdef khint_t k
    cdef size_t res = 0
    for i in range(n):
        k = kh_get_pyobjectmap(map.table,<pyobject_t> keys[i])
        if k != map.table.n_buckets:
            vals[i] = <object>map.table.vals[k]
            res += 1
        else:
            vals[i] = default_value
            if stop_at_unknown:
                return res
    return res

cpdef void swap_pyobjectmap(PyObjectMap a, PyObjectMap b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef kh_pyobjectmap_t *tmp=a.table
    a.table=b.table
    b.table=tmp


cpdef PyObjectMap copy_pyobjectmap(PyObjectMap s):
    if s is None:
        return None
    cdef PyObjectMap result = PyObjectMap(number_of_elements_hint=s.size())
    cdef PyObjectMapIterator it=s.get_iter(2)
    cdef pyobject_key_val_pair p
    while it.has_next():
        p = it.next()
        result.cput(<object>p.key, <object>p.val)
    return result


cpdef bint are_equal_pyobjectmap(PyObjectMap a, PyObjectMap b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    if a.size()!=b.size():
        return False
    cdef PyObjectMapIterator it=a.get_iter(2)
    cdef pyobject_key_val_pair p
    while it.has_next():
        p = it.next()
        if not b.contains(p.key):
            return False
    return True


cpdef void update_pyobjectmap(PyObjectMap a, PyObjectMap b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")
    cdef PyObjectMapIterator it=b.get_iter(2)
    cdef pyobject_key_val_pair p
    while it.has_next():
        p = it.next()
        a.cput(<object>p.key, <object>p.val)
