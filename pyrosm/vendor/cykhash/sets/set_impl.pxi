
"""
Template for sets

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""


from cpython.ref cimport Py_INCREF,Py_DECREF




#### special for PyObject:


cdef void _dealloc_int64(kh_int64set_t *table) nogil:
    if table is not NULL:
        kh_destroy_int64set(table)

cdef bint _contains_int64(kh_int64set_t *table, int64_t key) nogil:
    cdef khint_t k
    k = kh_get_int64set(table, key)
    return k != table.n_buckets

cdef void _add_int64(kh_int64set_t *table, int64_t key) nogil:
    cdef:
        khint_t k
        int ret = 0

    k = kh_put_int64set(table, key, &ret)
    table.keys[k] = key

cdef void _discard_int64(kh_int64set_t *table, int64_t key) nogil:
    cdef khint_t k
    k = kh_get_int64set(table, key)
    if k != table.n_buckets:
        kh_del_int64set(table, k)


### Iterator:
cdef class Int64SetIterator:

    cdef void __move(self) except *:
        while self.it<self.parent.table.n_buckets and not kh_exist_int64set(self.parent.table, self.it):
              self.it+=1       

    cdef bint has_next(self) except *:
        self.__move()
        return self.it < self.parent.table.n_buckets
      
    # doesn't work if there was change between last has_next() and next()  
    cdef int64_t next(self) except *:
        cdef int64_t result = self.parent.table.keys[self.it]
        self.it+=1#ensure at least one move!
        return result


    def __cinit__(self, Int64Set parent):
        self.parent = parent
        #search the start:
        self.it = 0
        self.__move()

    def __next__(self):
        if self.has_next():
            return self.next()
        else:
            raise StopIteration

#### the same for all:

cdef class Int64Set:

    def __cinit__(self, iterable=None, *, number_of_elements_hint=None):
        """
        iterable - initial elements in the set
        number_of_elements_hint - number of elements without the need of reallocation.
        """
        self.table = kh_init_int64set()
        if number_of_elements_hint is not None:    
            kh_resize_int64set(self.table, element_n_to_bucket_n(number_of_elements_hint))
        cdef int64_t el
        if iterable is not None:
            for el in iterable:
                self.add(el)

    def __len__(self):
        return self.size()
  
    cdef khint_t size(self):
        return self.table.size
        

    def __dealloc__(self):
        _dealloc_int64(self.table)
        self.table = NULL

    def __contains__(self, int64_t key):
        return self.contains(key)


    cdef bint contains(self, int64_t key) except *:
        return _contains_int64(self.table, key)


    cpdef void add(self, int64_t key) except *:
        _add_int64(self.table, key)

    
    cpdef void discard(self, int64_t key) except *:
        _discard_int64(self.table, key)


    cdef Int64SetIterator get_iter(self):
        return Int64SetIterator(self)

    def __iter__(self):
        return self.get_iter()

    def get_state_info(self):
        """
        returns information about state of the set

        >>> from cykhash import Int64Set
        >>> info = Int64Set([1]).get_state_info()
        >>> info["n_buckets"]
        4
        >>> info["n_occupied"]
        1

        """
        return {"n_buckets" : self.table.n_buckets, 
                "n_occupied" : self.table.n_occupied, 
                "upper_bound" : self.table.upper_bound}

    ### drop-in for set:
    def isdisjoint(self, other):
        if isinstance(other, Int64Set):
            return aredisjoint_int64(self, other)
        cdef int64_t el
        for el in other:
            if self.contains(el):
                return False
        return True

    def issuperset(self, other):
        if isinstance(other, Int64Set):
            return issubset_int64(self, other)
        cdef int64_t el
        for el in other:
            if not self.contains(el):
                return False
        return True

    def issubset(self, other):
        if isinstance(other, Int64Set):
            return issubset_int64(other, self)
        cdef int64_t el
        cdef Int64Set mem=Int64Set()
        for el in other:
            if self.contains(el):
                mem.add(el)
        return mem.size()==self.size()

    def __repr__(self):
        return "{"+','.join(map(str, self))+"}"

    def __le__(self, Int64Set other):
        return issubset_int64(other, self)

    def __lt__(self, Int64Set other):
        return issubset_int64(other, self) and self.size()<other.size()

    def __ge__(self, Int64Set other):
        return issubset_int64(self,  other)

    def __gt__(self, Int64Set other):
        return issubset_int64(self, other) and self.size()>other.size()

    def __eq__(self, Int64Set other):
        return issubset_int64(self, other) and self.size()==other.size()

    def __or__(self, Int64Set other):
        cdef Int64Set res = copy_int64(self)
        update_int64(res, other)
        return res

    def __ior__(self, Int64Set other):
        update_int64(self, other)
        return self

    def __and__(self, Int64Set other):
        return intersect_int64(self, other)

    def __iand__(self, Int64Set other):
        cdef Int64Set res = intersect_int64(self, other)
        swap_int64(self, res)
        return self

    def __sub__(self, Int64Set other):
        return difference_int64(self, other)

    def __isub__(self, Int64Set other):
        cdef Int64Set res = difference_int64(self, other)
        swap_int64(self, res)
        return self

    def __xor__(self, Int64Set other):
        return symmetric_difference_int64(self, other)

    def __ixor__(self, Int64Set other):
        cdef Int64Set res = symmetric_difference_int64(self, other)
        swap_int64(self, res)
        return self

    def copy(self):
        return copy_int64(self)

    def union(self, *others):
        cdef Int64Set res = copy_int64(self)
        for o in others:
            res.update(o)
        return res

    def update(self, other):
        if isinstance(other, Int64Set):
            update_int64(self, other)
            return
        cdef int64_t el
        for el in other:
            self.add(el)

    def intersection(self, *others):
        cdef Int64Set res = copy_int64(self)
        for o in others:
            res.intersection_update(o)
        return res

    def intersection_update(self, other):
        cdef Int64Set res 
        cdef int64_t el
        if isinstance(other, Int64Set):
            res = intersect_int64(self, other)
        else:
            res = Int64Set()
            for el in other:
                if self.contains(el):
                    res.add(el)
        swap_int64(self, res)

    def difference_update(self, other):
        cdef Int64Set res 
        cdef int64_t el
        if isinstance(other, Int64Set):
            res = difference_int64(self, other)
            swap_int64(self, res)
        else:
            for el in other:
                self.discard(el)

    def difference(self, *others):
        cdef Int64Set res = copy_int64(self)
        for o in others:
            res.difference_update(o)
        return res

    def symmetric_difference_update(self, other):
        cdef Int64Set res 
        cdef int64_t el
        if isinstance(other, Int64Set):
            res = symmetric_difference_int64(self, other)
        else:
            res = self.copy()
            for el in other:
                if self.contains(el):
                    res.discard(el)
                else:
                    res.add(el)
        swap_int64(self, res)

    def symmetric_difference(self, *others):
        cdef Int64Set res = copy_int64(self)
        for o in others:
            res.symmetric_difference_update(o)
        return res

    def clear(self):
        cdef Int64Set res = Int64Set()
        swap_int64(self, res)

    def remove(self, key):
        cdef size_t old=self.size()
        self.discard(key)
        if old==self.size():
            raise KeyError(key)

    def pop(self):
        if self.size()== 0:
            raise KeyError("pop from empty set")
        cdef Int64SetIterator it = self.get_iter()
        cdef int64_t el = it.next()
        self.discard(el)
        return el
        



### Utils:

def Int64Set_from(it):
    """
        creates Int64Set from an iterator. 
        Use Int64Set_from_buffer for a faster version if iterator is buffer of correct type
    """
    res=Int64Set()
    for i in it:
        res.add(i)
    return res

cpdef Int64Set Int64Set_from_buffer(int64_t[:] buf, double size_hint=0.0):
    """
        creates Int64Set from the given buffer buf. 
        Use slower Int64Set_from if series is given as iterator without buffer protocol.
        size_hint is an estimation of the ratio of unique elements. The default value of 0.0 means all elements in buf are expected to be unique
        Giving a good estimate will avoid rehashing (if estimate is too low) and having too big table (if estimate is too high).
    """
    cdef Py_ssize_t n = len(buf)
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Int64Set(number_of_elements_hint=at_least_needed)
    cdef Py_ssize_t i
    for i in range(n):
        res.add(buf[i])
    return res
    

cpdef void isin_int64(int64_t[:] query, Int64Set db, uint8_t[:] result) except *:
    """
        given query, db writes for every element of query True/False into result depending on whether query-element is in db (=True) or not (=False). 
        result should have the same length as query.
    """
    cdef size_t i
    cdef size_t n=len(query)
    if n!=len(result):
        raise ValueError("Different sizes for query({n}) and result({m})".format(n=n, m=len(result)))
    for i in range(n):
        result[i]=db is not None and db.contains(query[i])

cpdef bint all_int64(int64_t[:] query, Int64Set db) except *:
    """
        True if all elements of query are in db, False otherwise.
    """
    if query is None:
        return True
    cdef size_t i
    cdef size_t n=len(query)
    if db is None:
        return n==0
    for i in range(n):
        if not db.contains(query[i]):
            return False
    return True

cpdef bint all_int64_from_iter(object query, Int64Set db) except *:
    """
        True if all elements of query (as iterator) are in db, False otherwise.
    """
    if query is None:
        return True
    cdef int64_t el
    for el in query:
        if db is None or not db.contains(el):
            return False
    return True

cpdef bint none_int64(int64_t[:] query, Int64Set db) except *:
    """
        True if none of elements in query is in db, False otherwise.
    """
    if query is None or db is None:
        return True
    cdef size_t i
    cdef size_t n=len(query)
    for i in range(n):
        if db.contains(query[i]):
            return False
    return True

cpdef bint none_int64_from_iter(object query, Int64Set db) except *:
    """
        True if none of elements in query (as iterator) is in db, False otherwise.
    """
    if query is None or db is None:
        return True
    cdef int64_t el
    for el in query:
        if db.contains(el):
            return False
    return True

cpdef bint any_int64(int64_t[:] query, Int64Set db) except *:
    """
        True if one of elements in query is in db, False otherwise.
    """
    return not none_int64(query, db)

cpdef bint any_int64_from_iter(object query, Int64Set db) except *:
    """
        True if one of elements in query (as iterator) is in db, False otherwise.
    """
    return not none_int64_from_iter(query, db)

cpdef size_t count_if_int64(int64_t[:] query, Int64Set db) except *:
    """
        returns the number of (non-unique) elements in query, which are also in db
    """
    if query is None or db is None:
        return 0
    cdef size_t i
    cdef size_t n=len(query)
    cdef size_t res=0
    for i in range(n):
        if db.contains(query[i]):
            res+=1
    return res

cpdef size_t count_if_int64_from_iter(object query, Int64Set db) except *:
    """
        returns the number of (non-unique) elements in query (as iter), which are also in db
    """
    if query is None or db is None:
        return 0
    cdef int64_t el
    cdef size_t res=0
    for el in query:
        if db.contains(el):
            res+=1
    return res

cpdef bint aredisjoint_int64(Int64Set a, Int64Set b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef Int64SetIterator it
    cdef Int64Set s
    cdef int64_t el
    if a.size()<b.size():
        it=a.get_iter()
        s =b
    else:
        it=b.get_iter()
        s =a
    while it.has_next():
        el = it.next()
        if s.contains(el):
            return False
    return True

cpdef Int64Set intersect_int64(Int64Set a, Int64Set b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef Int64Set result = Int64Set()
    cdef Int64SetIterator it
    cdef Int64Set s
    cdef int64_t el
    if a.size()<b.size():
        it=a.get_iter()
        s =b
    else:
        it=b.get_iter()
        s =a
    while it.has_next():
        el = it.next()
        if s.contains(el):
            result.add(el)
    return result

cpdef bint issubset_int64(Int64Set s, Int64Set sub) except *:
    if s is None or sub is None:
        raise TypeError("'NoneType' object is not iterable")

    if s.size() < sub.size():
        return False

    cdef Int64SetIterator it=sub.get_iter()
    cdef int64_t el
    while it.has_next():
        el = it.next()
        if not s.contains(el):
            return False
    return True

cpdef Int64Set copy_int64(Int64Set s):
    if s is None:
        return None
    cdef Int64Set result = Int64Set(number_of_elements_hint=s.size())
    cdef Int64SetIterator it=s.get_iter()
    cdef int64_t el
    while it.has_next():
        el = it.next()
        result.add(el)
    return result

cpdef void update_int64(Int64Set s, Int64Set other) except *:
    if s is None or other is None:
        raise TypeError("'NoneType' object is not iterable")
    cdef Int64SetIterator it=other.get_iter()
    cdef int64_t el
    while it.has_next():
        el = it.next()
        s.add(el)

cpdef void swap_int64(Int64Set a, Int64Set b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef kh_int64set_t *tmp=a.table
    a.table=b.table
    b.table=tmp

cpdef Int64Set difference_int64(Int64Set a, Int64Set b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef int64_t el
    cdef Int64Set result = Int64Set()
    cdef Int64SetIterator it = a.get_iter()
    while it.has_next():
        el = it.next()
        if not b.contains(el):
            result.add(el)
    return result


cpdef Int64Set symmetric_difference_int64(Int64Set a, Int64Set b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef int64_t el
    cdef Int64Set result = Int64Set()
    cdef Int64SetIterator it = a.get_iter()
    while it.has_next():
          el = it.next()
          if not b.contains(el):
                result.add(el)
    it = b.get_iter()
    while it.has_next():
        el = it.next()
        if not a.contains(el):
            result.add(el)
    return result


#### special for PyObject:


cdef void _dealloc_float64(kh_float64set_t *table) nogil:
    if table is not NULL:
        kh_destroy_float64set(table)

cdef bint _contains_float64(kh_float64set_t *table, float64_t key) nogil:
    cdef khint_t k
    k = kh_get_float64set(table, key)
    return k != table.n_buckets

cdef void _add_float64(kh_float64set_t *table, float64_t key) nogil:
    cdef:
        khint_t k
        int ret = 0

    k = kh_put_float64set(table, key, &ret)
    table.keys[k] = key

cdef void _discard_float64(kh_float64set_t *table, float64_t key) nogil:
    cdef khint_t k
    k = kh_get_float64set(table, key)
    if k != table.n_buckets:
        kh_del_float64set(table, k)


### Iterator:
cdef class Float64SetIterator:

    cdef void __move(self) except *:
        while self.it<self.parent.table.n_buckets and not kh_exist_float64set(self.parent.table, self.it):
              self.it+=1       

    cdef bint has_next(self) except *:
        self.__move()
        return self.it < self.parent.table.n_buckets
      
    # doesn't work if there was change between last has_next() and next()  
    cdef float64_t next(self) except *:
        cdef float64_t result = self.parent.table.keys[self.it]
        self.it+=1#ensure at least one move!
        return result


    def __cinit__(self, Float64Set parent):
        self.parent = parent
        #search the start:
        self.it = 0
        self.__move()

    def __next__(self):
        if self.has_next():
            return self.next()
        else:
            raise StopIteration

#### the same for all:

cdef class Float64Set:

    def __cinit__(self, iterable=None, *, number_of_elements_hint=None):
        """
        iterable - initial elements in the set
        number_of_elements_hint - number of elements without the need of reallocation.
        """
        self.table = kh_init_float64set()
        if number_of_elements_hint is not None:    
            kh_resize_float64set(self.table, element_n_to_bucket_n(number_of_elements_hint))
        cdef float64_t el
        if iterable is not None:
            for el in iterable:
                self.add(el)

    def __len__(self):
        return self.size()
  
    cdef khint_t size(self):
        return self.table.size
        

    def __dealloc__(self):
        _dealloc_float64(self.table)
        self.table = NULL

    def __contains__(self, float64_t key):
        return self.contains(key)


    cdef bint contains(self, float64_t key) except *:
        return _contains_float64(self.table, key)


    cpdef void add(self, float64_t key) except *:
        _add_float64(self.table, key)

    
    cpdef void discard(self, float64_t key) except *:
        _discard_float64(self.table, key)


    cdef Float64SetIterator get_iter(self):
        return Float64SetIterator(self)

    def __iter__(self):
        return self.get_iter()

    def get_state_info(self):
        """
        returns information about state of the set

        >>> from cykhash import Float64Set
        >>> info = Float64Set([1]).get_state_info()
        >>> info["n_buckets"]
        4
        >>> info["n_occupied"]
        1

        """
        return {"n_buckets" : self.table.n_buckets, 
                "n_occupied" : self.table.n_occupied, 
                "upper_bound" : self.table.upper_bound}

    ### drop-in for set:
    def isdisjoint(self, other):
        if isinstance(other, Float64Set):
            return aredisjoint_float64(self, other)
        cdef float64_t el
        for el in other:
            if self.contains(el):
                return False
        return True

    def issuperset(self, other):
        if isinstance(other, Float64Set):
            return issubset_float64(self, other)
        cdef float64_t el
        for el in other:
            if not self.contains(el):
                return False
        return True

    def issubset(self, other):
        if isinstance(other, Float64Set):
            return issubset_float64(other, self)
        cdef float64_t el
        cdef Float64Set mem=Float64Set()
        for el in other:
            if self.contains(el):
                mem.add(el)
        return mem.size()==self.size()

    def __repr__(self):
        return "{"+','.join(map(str, self))+"}"

    def __le__(self, Float64Set other):
        return issubset_float64(other, self)

    def __lt__(self, Float64Set other):
        return issubset_float64(other, self) and self.size()<other.size()

    def __ge__(self, Float64Set other):
        return issubset_float64(self,  other)

    def __gt__(self, Float64Set other):
        return issubset_float64(self, other) and self.size()>other.size()

    def __eq__(self, Float64Set other):
        return issubset_float64(self, other) and self.size()==other.size()

    def __or__(self, Float64Set other):
        cdef Float64Set res = copy_float64(self)
        update_float64(res, other)
        return res

    def __ior__(self, Float64Set other):
        update_float64(self, other)
        return self

    def __and__(self, Float64Set other):
        return intersect_float64(self, other)

    def __iand__(self, Float64Set other):
        cdef Float64Set res = intersect_float64(self, other)
        swap_float64(self, res)
        return self

    def __sub__(self, Float64Set other):
        return difference_float64(self, other)

    def __isub__(self, Float64Set other):
        cdef Float64Set res = difference_float64(self, other)
        swap_float64(self, res)
        return self

    def __xor__(self, Float64Set other):
        return symmetric_difference_float64(self, other)

    def __ixor__(self, Float64Set other):
        cdef Float64Set res = symmetric_difference_float64(self, other)
        swap_float64(self, res)
        return self

    def copy(self):
        return copy_float64(self)

    def union(self, *others):
        cdef Float64Set res = copy_float64(self)
        for o in others:
            res.update(o)
        return res

    def update(self, other):
        if isinstance(other, Float64Set):
            update_float64(self, other)
            return
        cdef float64_t el
        for el in other:
            self.add(el)

    def intersection(self, *others):
        cdef Float64Set res = copy_float64(self)
        for o in others:
            res.intersection_update(o)
        return res

    def intersection_update(self, other):
        cdef Float64Set res 
        cdef float64_t el
        if isinstance(other, Float64Set):
            res = intersect_float64(self, other)
        else:
            res = Float64Set()
            for el in other:
                if self.contains(el):
                    res.add(el)
        swap_float64(self, res)

    def difference_update(self, other):
        cdef Float64Set res 
        cdef float64_t el
        if isinstance(other, Float64Set):
            res = difference_float64(self, other)
            swap_float64(self, res)
        else:
            for el in other:
                self.discard(el)

    def difference(self, *others):
        cdef Float64Set res = copy_float64(self)
        for o in others:
            res.difference_update(o)
        return res

    def symmetric_difference_update(self, other):
        cdef Float64Set res 
        cdef float64_t el
        if isinstance(other, Float64Set):
            res = symmetric_difference_float64(self, other)
        else:
            res = self.copy()
            for el in other:
                if self.contains(el):
                    res.discard(el)
                else:
                    res.add(el)
        swap_float64(self, res)

    def symmetric_difference(self, *others):
        cdef Float64Set res = copy_float64(self)
        for o in others:
            res.symmetric_difference_update(o)
        return res

    def clear(self):
        cdef Float64Set res = Float64Set()
        swap_float64(self, res)

    def remove(self, key):
        cdef size_t old=self.size()
        self.discard(key)
        if old==self.size():
            raise KeyError(key)

    def pop(self):
        if self.size()== 0:
            raise KeyError("pop from empty set")
        cdef Float64SetIterator it = self.get_iter()
        cdef float64_t el = it.next()
        self.discard(el)
        return el
        



### Utils:

def Float64Set_from(it):
    """
        creates Float64Set from an iterator. 
        Use Float64Set_from_buffer for a faster version if iterator is buffer of correct type
    """
    res=Float64Set()
    for i in it:
        res.add(i)
    return res

cpdef Float64Set Float64Set_from_buffer(float64_t[:] buf, double size_hint=0.0):
    """
        creates Float64Set from the given buffer buf. 
        Use slower Float64Set_from if series is given as iterator without buffer protocol.
        size_hint is an estimation of the ratio of unique elements. The default value of 0.0 means all elements in buf are expected to be unique
        Giving a good estimate will avoid rehashing (if estimate is too low) and having too big table (if estimate is too high).
    """
    cdef Py_ssize_t n = len(buf)
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Float64Set(number_of_elements_hint=at_least_needed)
    cdef Py_ssize_t i
    for i in range(n):
        res.add(buf[i])
    return res
    

cpdef void isin_float64(float64_t[:] query, Float64Set db, uint8_t[:] result) except *:
    """
        given query, db writes for every element of query True/False into result depending on whether query-element is in db (=True) or not (=False). 
        result should have the same length as query.
    """
    cdef size_t i
    cdef size_t n=len(query)
    if n!=len(result):
        raise ValueError("Different sizes for query({n}) and result({m})".format(n=n, m=len(result)))
    for i in range(n):
        result[i]=db is not None and db.contains(query[i])

cpdef bint all_float64(float64_t[:] query, Float64Set db) except *:
    """
        True if all elements of query are in db, False otherwise.
    """
    if query is None:
        return True
    cdef size_t i
    cdef size_t n=len(query)
    if db is None:
        return n==0
    for i in range(n):
        if not db.contains(query[i]):
            return False
    return True

cpdef bint all_float64_from_iter(object query, Float64Set db) except *:
    """
        True if all elements of query (as iterator) are in db, False otherwise.
    """
    if query is None:
        return True
    cdef float64_t el
    for el in query:
        if db is None or not db.contains(el):
            return False
    return True

cpdef bint none_float64(float64_t[:] query, Float64Set db) except *:
    """
        True if none of elements in query is in db, False otherwise.
    """
    if query is None or db is None:
        return True
    cdef size_t i
    cdef size_t n=len(query)
    for i in range(n):
        if db.contains(query[i]):
            return False
    return True

cpdef bint none_float64_from_iter(object query, Float64Set db) except *:
    """
        True if none of elements in query (as iterator) is in db, False otherwise.
    """
    if query is None or db is None:
        return True
    cdef float64_t el
    for el in query:
        if db.contains(el):
            return False
    return True

cpdef bint any_float64(float64_t[:] query, Float64Set db) except *:
    """
        True if one of elements in query is in db, False otherwise.
    """
    return not none_float64(query, db)

cpdef bint any_float64_from_iter(object query, Float64Set db) except *:
    """
        True if one of elements in query (as iterator) is in db, False otherwise.
    """
    return not none_float64_from_iter(query, db)

cpdef size_t count_if_float64(float64_t[:] query, Float64Set db) except *:
    """
        returns the number of (non-unique) elements in query, which are also in db
    """
    if query is None or db is None:
        return 0
    cdef size_t i
    cdef size_t n=len(query)
    cdef size_t res=0
    for i in range(n):
        if db.contains(query[i]):
            res+=1
    return res

cpdef size_t count_if_float64_from_iter(object query, Float64Set db) except *:
    """
        returns the number of (non-unique) elements in query (as iter), which are also in db
    """
    if query is None or db is None:
        return 0
    cdef float64_t el
    cdef size_t res=0
    for el in query:
        if db.contains(el):
            res+=1
    return res

cpdef bint aredisjoint_float64(Float64Set a, Float64Set b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef Float64SetIterator it
    cdef Float64Set s
    cdef float64_t el
    if a.size()<b.size():
        it=a.get_iter()
        s =b
    else:
        it=b.get_iter()
        s =a
    while it.has_next():
        el = it.next()
        if s.contains(el):
            return False
    return True

cpdef Float64Set intersect_float64(Float64Set a, Float64Set b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef Float64Set result = Float64Set()
    cdef Float64SetIterator it
    cdef Float64Set s
    cdef float64_t el
    if a.size()<b.size():
        it=a.get_iter()
        s =b
    else:
        it=b.get_iter()
        s =a
    while it.has_next():
        el = it.next()
        if s.contains(el):
            result.add(el)
    return result

cpdef bint issubset_float64(Float64Set s, Float64Set sub) except *:
    if s is None or sub is None:
        raise TypeError("'NoneType' object is not iterable")

    if s.size() < sub.size():
        return False

    cdef Float64SetIterator it=sub.get_iter()
    cdef float64_t el
    while it.has_next():
        el = it.next()
        if not s.contains(el):
            return False
    return True

cpdef Float64Set copy_float64(Float64Set s):
    if s is None:
        return None
    cdef Float64Set result = Float64Set(number_of_elements_hint=s.size())
    cdef Float64SetIterator it=s.get_iter()
    cdef float64_t el
    while it.has_next():
        el = it.next()
        result.add(el)
    return result

cpdef void update_float64(Float64Set s, Float64Set other) except *:
    if s is None or other is None:
        raise TypeError("'NoneType' object is not iterable")
    cdef Float64SetIterator it=other.get_iter()
    cdef float64_t el
    while it.has_next():
        el = it.next()
        s.add(el)

cpdef void swap_float64(Float64Set a, Float64Set b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef kh_float64set_t *tmp=a.table
    a.table=b.table
    b.table=tmp

cpdef Float64Set difference_float64(Float64Set a, Float64Set b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef float64_t el
    cdef Float64Set result = Float64Set()
    cdef Float64SetIterator it = a.get_iter()
    while it.has_next():
        el = it.next()
        if not b.contains(el):
            result.add(el)
    return result


cpdef Float64Set symmetric_difference_float64(Float64Set a, Float64Set b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef float64_t el
    cdef Float64Set result = Float64Set()
    cdef Float64SetIterator it = a.get_iter()
    while it.has_next():
          el = it.next()
          if not b.contains(el):
                result.add(el)
    it = b.get_iter()
    while it.has_next():
        el = it.next()
        if not a.contains(el):
            result.add(el)
    return result


#### special for PyObject:


cdef void _dealloc_int32(kh_int32set_t *table) nogil:
    if table is not NULL:
        kh_destroy_int32set(table)

cdef bint _contains_int32(kh_int32set_t *table, int32_t key) nogil:
    cdef khint_t k
    k = kh_get_int32set(table, key)
    return k != table.n_buckets

cdef void _add_int32(kh_int32set_t *table, int32_t key) nogil:
    cdef:
        khint_t k
        int ret = 0

    k = kh_put_int32set(table, key, &ret)
    table.keys[k] = key

cdef void _discard_int32(kh_int32set_t *table, int32_t key) nogil:
    cdef khint_t k
    k = kh_get_int32set(table, key)
    if k != table.n_buckets:
        kh_del_int32set(table, k)


### Iterator:
cdef class Int32SetIterator:

    cdef void __move(self) except *:
        while self.it<self.parent.table.n_buckets and not kh_exist_int32set(self.parent.table, self.it):
              self.it+=1       

    cdef bint has_next(self) except *:
        self.__move()
        return self.it < self.parent.table.n_buckets
      
    # doesn't work if there was change between last has_next() and next()  
    cdef int32_t next(self) except *:
        cdef int32_t result = self.parent.table.keys[self.it]
        self.it+=1#ensure at least one move!
        return result


    def __cinit__(self, Int32Set parent):
        self.parent = parent
        #search the start:
        self.it = 0
        self.__move()

    def __next__(self):
        if self.has_next():
            return self.next()
        else:
            raise StopIteration

#### the same for all:

cdef class Int32Set:

    def __cinit__(self, iterable=None, *, number_of_elements_hint=None):
        """
        iterable - initial elements in the set
        number_of_elements_hint - number of elements without the need of reallocation.
        """
        self.table = kh_init_int32set()
        if number_of_elements_hint is not None:    
            kh_resize_int32set(self.table, element_n_to_bucket_n(number_of_elements_hint))
        cdef int32_t el
        if iterable is not None:
            for el in iterable:
                self.add(el)

    def __len__(self):
        return self.size()
  
    cdef khint_t size(self):
        return self.table.size
        

    def __dealloc__(self):
        _dealloc_int32(self.table)
        self.table = NULL

    def __contains__(self, int32_t key):
        return self.contains(key)


    cdef bint contains(self, int32_t key) except *:
        return _contains_int32(self.table, key)


    cpdef void add(self, int32_t key) except *:
        _add_int32(self.table, key)

    
    cpdef void discard(self, int32_t key) except *:
        _discard_int32(self.table, key)


    cdef Int32SetIterator get_iter(self):
        return Int32SetIterator(self)

    def __iter__(self):
        return self.get_iter()

    def get_state_info(self):
        """
        returns information about state of the set

        >>> from cykhash import Int32Set
        >>> info = Int32Set([1]).get_state_info()
        >>> info["n_buckets"]
        4
        >>> info["n_occupied"]
        1

        """
        return {"n_buckets" : self.table.n_buckets, 
                "n_occupied" : self.table.n_occupied, 
                "upper_bound" : self.table.upper_bound}

    ### drop-in for set:
    def isdisjoint(self, other):
        if isinstance(other, Int32Set):
            return aredisjoint_int32(self, other)
        cdef int32_t el
        for el in other:
            if self.contains(el):
                return False
        return True

    def issuperset(self, other):
        if isinstance(other, Int32Set):
            return issubset_int32(self, other)
        cdef int32_t el
        for el in other:
            if not self.contains(el):
                return False
        return True

    def issubset(self, other):
        if isinstance(other, Int32Set):
            return issubset_int32(other, self)
        cdef int32_t el
        cdef Int32Set mem=Int32Set()
        for el in other:
            if self.contains(el):
                mem.add(el)
        return mem.size()==self.size()

    def __repr__(self):
        return "{"+','.join(map(str, self))+"}"

    def __le__(self, Int32Set other):
        return issubset_int32(other, self)

    def __lt__(self, Int32Set other):
        return issubset_int32(other, self) and self.size()<other.size()

    def __ge__(self, Int32Set other):
        return issubset_int32(self,  other)

    def __gt__(self, Int32Set other):
        return issubset_int32(self, other) and self.size()>other.size()

    def __eq__(self, Int32Set other):
        return issubset_int32(self, other) and self.size()==other.size()

    def __or__(self, Int32Set other):
        cdef Int32Set res = copy_int32(self)
        update_int32(res, other)
        return res

    def __ior__(self, Int32Set other):
        update_int32(self, other)
        return self

    def __and__(self, Int32Set other):
        return intersect_int32(self, other)

    def __iand__(self, Int32Set other):
        cdef Int32Set res = intersect_int32(self, other)
        swap_int32(self, res)
        return self

    def __sub__(self, Int32Set other):
        return difference_int32(self, other)

    def __isub__(self, Int32Set other):
        cdef Int32Set res = difference_int32(self, other)
        swap_int32(self, res)
        return self

    def __xor__(self, Int32Set other):
        return symmetric_difference_int32(self, other)

    def __ixor__(self, Int32Set other):
        cdef Int32Set res = symmetric_difference_int32(self, other)
        swap_int32(self, res)
        return self

    def copy(self):
        return copy_int32(self)

    def union(self, *others):
        cdef Int32Set res = copy_int32(self)
        for o in others:
            res.update(o)
        return res

    def update(self, other):
        if isinstance(other, Int32Set):
            update_int32(self, other)
            return
        cdef int32_t el
        for el in other:
            self.add(el)

    def intersection(self, *others):
        cdef Int32Set res = copy_int32(self)
        for o in others:
            res.intersection_update(o)
        return res

    def intersection_update(self, other):
        cdef Int32Set res 
        cdef int32_t el
        if isinstance(other, Int32Set):
            res = intersect_int32(self, other)
        else:
            res = Int32Set()
            for el in other:
                if self.contains(el):
                    res.add(el)
        swap_int32(self, res)

    def difference_update(self, other):
        cdef Int32Set res 
        cdef int32_t el
        if isinstance(other, Int32Set):
            res = difference_int32(self, other)
            swap_int32(self, res)
        else:
            for el in other:
                self.discard(el)

    def difference(self, *others):
        cdef Int32Set res = copy_int32(self)
        for o in others:
            res.difference_update(o)
        return res

    def symmetric_difference_update(self, other):
        cdef Int32Set res 
        cdef int32_t el
        if isinstance(other, Int32Set):
            res = symmetric_difference_int32(self, other)
        else:
            res = self.copy()
            for el in other:
                if self.contains(el):
                    res.discard(el)
                else:
                    res.add(el)
        swap_int32(self, res)

    def symmetric_difference(self, *others):
        cdef Int32Set res = copy_int32(self)
        for o in others:
            res.symmetric_difference_update(o)
        return res

    def clear(self):
        cdef Int32Set res = Int32Set()
        swap_int32(self, res)

    def remove(self, key):
        cdef size_t old=self.size()
        self.discard(key)
        if old==self.size():
            raise KeyError(key)

    def pop(self):
        if self.size()== 0:
            raise KeyError("pop from empty set")
        cdef Int32SetIterator it = self.get_iter()
        cdef int32_t el = it.next()
        self.discard(el)
        return el
        



### Utils:

def Int32Set_from(it):
    """
        creates Int32Set from an iterator. 
        Use Int32Set_from_buffer for a faster version if iterator is buffer of correct type
    """
    res=Int32Set()
    for i in it:
        res.add(i)
    return res

cpdef Int32Set Int32Set_from_buffer(int32_t[:] buf, double size_hint=0.0):
    """
        creates Int32Set from the given buffer buf. 
        Use slower Int32Set_from if series is given as iterator without buffer protocol.
        size_hint is an estimation of the ratio of unique elements. The default value of 0.0 means all elements in buf are expected to be unique
        Giving a good estimate will avoid rehashing (if estimate is too low) and having too big table (if estimate is too high).
    """
    cdef Py_ssize_t n = len(buf)
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Int32Set(number_of_elements_hint=at_least_needed)
    cdef Py_ssize_t i
    for i in range(n):
        res.add(buf[i])
    return res
    

cpdef void isin_int32(int32_t[:] query, Int32Set db, uint8_t[:] result) except *:
    """
        given query, db writes for every element of query True/False into result depending on whether query-element is in db (=True) or not (=False). 
        result should have the same length as query.
    """
    cdef size_t i
    cdef size_t n=len(query)
    if n!=len(result):
        raise ValueError("Different sizes for query({n}) and result({m})".format(n=n, m=len(result)))
    for i in range(n):
        result[i]=db is not None and db.contains(query[i])

cpdef bint all_int32(int32_t[:] query, Int32Set db) except *:
    """
        True if all elements of query are in db, False otherwise.
    """
    if query is None:
        return True
    cdef size_t i
    cdef size_t n=len(query)
    if db is None:
        return n==0
    for i in range(n):
        if not db.contains(query[i]):
            return False
    return True

cpdef bint all_int32_from_iter(object query, Int32Set db) except *:
    """
        True if all elements of query (as iterator) are in db, False otherwise.
    """
    if query is None:
        return True
    cdef int32_t el
    for el in query:
        if db is None or not db.contains(el):
            return False
    return True

cpdef bint none_int32(int32_t[:] query, Int32Set db) except *:
    """
        True if none of elements in query is in db, False otherwise.
    """
    if query is None or db is None:
        return True
    cdef size_t i
    cdef size_t n=len(query)
    for i in range(n):
        if db.contains(query[i]):
            return False
    return True

cpdef bint none_int32_from_iter(object query, Int32Set db) except *:
    """
        True if none of elements in query (as iterator) is in db, False otherwise.
    """
    if query is None or db is None:
        return True
    cdef int32_t el
    for el in query:
        if db.contains(el):
            return False
    return True

cpdef bint any_int32(int32_t[:] query, Int32Set db) except *:
    """
        True if one of elements in query is in db, False otherwise.
    """
    return not none_int32(query, db)

cpdef bint any_int32_from_iter(object query, Int32Set db) except *:
    """
        True if one of elements in query (as iterator) is in db, False otherwise.
    """
    return not none_int32_from_iter(query, db)

cpdef size_t count_if_int32(int32_t[:] query, Int32Set db) except *:
    """
        returns the number of (non-unique) elements in query, which are also in db
    """
    if query is None or db is None:
        return 0
    cdef size_t i
    cdef size_t n=len(query)
    cdef size_t res=0
    for i in range(n):
        if db.contains(query[i]):
            res+=1
    return res

cpdef size_t count_if_int32_from_iter(object query, Int32Set db) except *:
    """
        returns the number of (non-unique) elements in query (as iter), which are also in db
    """
    if query is None or db is None:
        return 0
    cdef int32_t el
    cdef size_t res=0
    for el in query:
        if db.contains(el):
            res+=1
    return res

cpdef bint aredisjoint_int32(Int32Set a, Int32Set b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef Int32SetIterator it
    cdef Int32Set s
    cdef int32_t el
    if a.size()<b.size():
        it=a.get_iter()
        s =b
    else:
        it=b.get_iter()
        s =a
    while it.has_next():
        el = it.next()
        if s.contains(el):
            return False
    return True

cpdef Int32Set intersect_int32(Int32Set a, Int32Set b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef Int32Set result = Int32Set()
    cdef Int32SetIterator it
    cdef Int32Set s
    cdef int32_t el
    if a.size()<b.size():
        it=a.get_iter()
        s =b
    else:
        it=b.get_iter()
        s =a
    while it.has_next():
        el = it.next()
        if s.contains(el):
            result.add(el)
    return result

cpdef bint issubset_int32(Int32Set s, Int32Set sub) except *:
    if s is None or sub is None:
        raise TypeError("'NoneType' object is not iterable")

    if s.size() < sub.size():
        return False

    cdef Int32SetIterator it=sub.get_iter()
    cdef int32_t el
    while it.has_next():
        el = it.next()
        if not s.contains(el):
            return False
    return True

cpdef Int32Set copy_int32(Int32Set s):
    if s is None:
        return None
    cdef Int32Set result = Int32Set(number_of_elements_hint=s.size())
    cdef Int32SetIterator it=s.get_iter()
    cdef int32_t el
    while it.has_next():
        el = it.next()
        result.add(el)
    return result

cpdef void update_int32(Int32Set s, Int32Set other) except *:
    if s is None or other is None:
        raise TypeError("'NoneType' object is not iterable")
    cdef Int32SetIterator it=other.get_iter()
    cdef int32_t el
    while it.has_next():
        el = it.next()
        s.add(el)

cpdef void swap_int32(Int32Set a, Int32Set b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef kh_int32set_t *tmp=a.table
    a.table=b.table
    b.table=tmp

cpdef Int32Set difference_int32(Int32Set a, Int32Set b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef int32_t el
    cdef Int32Set result = Int32Set()
    cdef Int32SetIterator it = a.get_iter()
    while it.has_next():
        el = it.next()
        if not b.contains(el):
            result.add(el)
    return result


cpdef Int32Set symmetric_difference_int32(Int32Set a, Int32Set b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef int32_t el
    cdef Int32Set result = Int32Set()
    cdef Int32SetIterator it = a.get_iter()
    while it.has_next():
          el = it.next()
          if not b.contains(el):
                result.add(el)
    it = b.get_iter()
    while it.has_next():
        el = it.next()
        if not a.contains(el):
            result.add(el)
    return result


#### special for PyObject:


cdef void _dealloc_float32(kh_float32set_t *table) nogil:
    if table is not NULL:
        kh_destroy_float32set(table)

cdef bint _contains_float32(kh_float32set_t *table, float32_t key) nogil:
    cdef khint_t k
    k = kh_get_float32set(table, key)
    return k != table.n_buckets

cdef void _add_float32(kh_float32set_t *table, float32_t key) nogil:
    cdef:
        khint_t k
        int ret = 0

    k = kh_put_float32set(table, key, &ret)
    table.keys[k] = key

cdef void _discard_float32(kh_float32set_t *table, float32_t key) nogil:
    cdef khint_t k
    k = kh_get_float32set(table, key)
    if k != table.n_buckets:
        kh_del_float32set(table, k)


### Iterator:
cdef class Float32SetIterator:

    cdef void __move(self) except *:
        while self.it<self.parent.table.n_buckets and not kh_exist_float32set(self.parent.table, self.it):
              self.it+=1       

    cdef bint has_next(self) except *:
        self.__move()
        return self.it < self.parent.table.n_buckets
      
    # doesn't work if there was change between last has_next() and next()  
    cdef float32_t next(self) except *:
        cdef float32_t result = self.parent.table.keys[self.it]
        self.it+=1#ensure at least one move!
        return result


    def __cinit__(self, Float32Set parent):
        self.parent = parent
        #search the start:
        self.it = 0
        self.__move()

    def __next__(self):
        if self.has_next():
            return self.next()
        else:
            raise StopIteration

#### the same for all:

cdef class Float32Set:

    def __cinit__(self, iterable=None, *, number_of_elements_hint=None):
        """
        iterable - initial elements in the set
        number_of_elements_hint - number of elements without the need of reallocation.
        """
        self.table = kh_init_float32set()
        if number_of_elements_hint is not None:    
            kh_resize_float32set(self.table, element_n_to_bucket_n(number_of_elements_hint))
        cdef float32_t el
        if iterable is not None:
            for el in iterable:
                self.add(el)

    def __len__(self):
        return self.size()
  
    cdef khint_t size(self):
        return self.table.size
        

    def __dealloc__(self):
        _dealloc_float32(self.table)
        self.table = NULL

    def __contains__(self, float32_t key):
        return self.contains(key)


    cdef bint contains(self, float32_t key) except *:
        return _contains_float32(self.table, key)


    cpdef void add(self, float32_t key) except *:
        _add_float32(self.table, key)

    
    cpdef void discard(self, float32_t key) except *:
        _discard_float32(self.table, key)


    cdef Float32SetIterator get_iter(self):
        return Float32SetIterator(self)

    def __iter__(self):
        return self.get_iter()

    def get_state_info(self):
        """
        returns information about state of the set

        >>> from cykhash import Float32Set
        >>> info = Float32Set([1]).get_state_info()
        >>> info["n_buckets"]
        4
        >>> info["n_occupied"]
        1

        """
        return {"n_buckets" : self.table.n_buckets, 
                "n_occupied" : self.table.n_occupied, 
                "upper_bound" : self.table.upper_bound}

    ### drop-in for set:
    def isdisjoint(self, other):
        if isinstance(other, Float32Set):
            return aredisjoint_float32(self, other)
        cdef float32_t el
        for el in other:
            if self.contains(el):
                return False
        return True

    def issuperset(self, other):
        if isinstance(other, Float32Set):
            return issubset_float32(self, other)
        cdef float32_t el
        for el in other:
            if not self.contains(el):
                return False
        return True

    def issubset(self, other):
        if isinstance(other, Float32Set):
            return issubset_float32(other, self)
        cdef float32_t el
        cdef Float32Set mem=Float32Set()
        for el in other:
            if self.contains(el):
                mem.add(el)
        return mem.size()==self.size()

    def __repr__(self):
        return "{"+','.join(map(str, self))+"}"

    def __le__(self, Float32Set other):
        return issubset_float32(other, self)

    def __lt__(self, Float32Set other):
        return issubset_float32(other, self) and self.size()<other.size()

    def __ge__(self, Float32Set other):
        return issubset_float32(self,  other)

    def __gt__(self, Float32Set other):
        return issubset_float32(self, other) and self.size()>other.size()

    def __eq__(self, Float32Set other):
        return issubset_float32(self, other) and self.size()==other.size()

    def __or__(self, Float32Set other):
        cdef Float32Set res = copy_float32(self)
        update_float32(res, other)
        return res

    def __ior__(self, Float32Set other):
        update_float32(self, other)
        return self

    def __and__(self, Float32Set other):
        return intersect_float32(self, other)

    def __iand__(self, Float32Set other):
        cdef Float32Set res = intersect_float32(self, other)
        swap_float32(self, res)
        return self

    def __sub__(self, Float32Set other):
        return difference_float32(self, other)

    def __isub__(self, Float32Set other):
        cdef Float32Set res = difference_float32(self, other)
        swap_float32(self, res)
        return self

    def __xor__(self, Float32Set other):
        return symmetric_difference_float32(self, other)

    def __ixor__(self, Float32Set other):
        cdef Float32Set res = symmetric_difference_float32(self, other)
        swap_float32(self, res)
        return self

    def copy(self):
        return copy_float32(self)

    def union(self, *others):
        cdef Float32Set res = copy_float32(self)
        for o in others:
            res.update(o)
        return res

    def update(self, other):
        if isinstance(other, Float32Set):
            update_float32(self, other)
            return
        cdef float32_t el
        for el in other:
            self.add(el)

    def intersection(self, *others):
        cdef Float32Set res = copy_float32(self)
        for o in others:
            res.intersection_update(o)
        return res

    def intersection_update(self, other):
        cdef Float32Set res 
        cdef float32_t el
        if isinstance(other, Float32Set):
            res = intersect_float32(self, other)
        else:
            res = Float32Set()
            for el in other:
                if self.contains(el):
                    res.add(el)
        swap_float32(self, res)

    def difference_update(self, other):
        cdef Float32Set res 
        cdef float32_t el
        if isinstance(other, Float32Set):
            res = difference_float32(self, other)
            swap_float32(self, res)
        else:
            for el in other:
                self.discard(el)

    def difference(self, *others):
        cdef Float32Set res = copy_float32(self)
        for o in others:
            res.difference_update(o)
        return res

    def symmetric_difference_update(self, other):
        cdef Float32Set res 
        cdef float32_t el
        if isinstance(other, Float32Set):
            res = symmetric_difference_float32(self, other)
        else:
            res = self.copy()
            for el in other:
                if self.contains(el):
                    res.discard(el)
                else:
                    res.add(el)
        swap_float32(self, res)

    def symmetric_difference(self, *others):
        cdef Float32Set res = copy_float32(self)
        for o in others:
            res.symmetric_difference_update(o)
        return res

    def clear(self):
        cdef Float32Set res = Float32Set()
        swap_float32(self, res)

    def remove(self, key):
        cdef size_t old=self.size()
        self.discard(key)
        if old==self.size():
            raise KeyError(key)

    def pop(self):
        if self.size()== 0:
            raise KeyError("pop from empty set")
        cdef Float32SetIterator it = self.get_iter()
        cdef float32_t el = it.next()
        self.discard(el)
        return el
        



### Utils:

def Float32Set_from(it):
    """
        creates Float32Set from an iterator. 
        Use Float32Set_from_buffer for a faster version if iterator is buffer of correct type
    """
    res=Float32Set()
    for i in it:
        res.add(i)
    return res

cpdef Float32Set Float32Set_from_buffer(float32_t[:] buf, double size_hint=0.0):
    """
        creates Float32Set from the given buffer buf. 
        Use slower Float32Set_from if series is given as iterator without buffer protocol.
        size_hint is an estimation of the ratio of unique elements. The default value of 0.0 means all elements in buf are expected to be unique
        Giving a good estimate will avoid rehashing (if estimate is too low) and having too big table (if estimate is too high).
    """
    cdef Py_ssize_t n = len(buf)
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Float32Set(number_of_elements_hint=at_least_needed)
    cdef Py_ssize_t i
    for i in range(n):
        res.add(buf[i])
    return res
    

cpdef void isin_float32(float32_t[:] query, Float32Set db, uint8_t[:] result) except *:
    """
        given query, db writes for every element of query True/False into result depending on whether query-element is in db (=True) or not (=False). 
        result should have the same length as query.
    """
    cdef size_t i
    cdef size_t n=len(query)
    if n!=len(result):
        raise ValueError("Different sizes for query({n}) and result({m})".format(n=n, m=len(result)))
    for i in range(n):
        result[i]=db is not None and db.contains(query[i])

cpdef bint all_float32(float32_t[:] query, Float32Set db) except *:
    """
        True if all elements of query are in db, False otherwise.
    """
    if query is None:
        return True
    cdef size_t i
    cdef size_t n=len(query)
    if db is None:
        return n==0
    for i in range(n):
        if not db.contains(query[i]):
            return False
    return True

cpdef bint all_float32_from_iter(object query, Float32Set db) except *:
    """
        True if all elements of query (as iterator) are in db, False otherwise.
    """
    if query is None:
        return True
    cdef float32_t el
    for el in query:
        if db is None or not db.contains(el):
            return False
    return True

cpdef bint none_float32(float32_t[:] query, Float32Set db) except *:
    """
        True if none of elements in query is in db, False otherwise.
    """
    if query is None or db is None:
        return True
    cdef size_t i
    cdef size_t n=len(query)
    for i in range(n):
        if db.contains(query[i]):
            return False
    return True

cpdef bint none_float32_from_iter(object query, Float32Set db) except *:
    """
        True if none of elements in query (as iterator) is in db, False otherwise.
    """
    if query is None or db is None:
        return True
    cdef float32_t el
    for el in query:
        if db.contains(el):
            return False
    return True

cpdef bint any_float32(float32_t[:] query, Float32Set db) except *:
    """
        True if one of elements in query is in db, False otherwise.
    """
    return not none_float32(query, db)

cpdef bint any_float32_from_iter(object query, Float32Set db) except *:
    """
        True if one of elements in query (as iterator) is in db, False otherwise.
    """
    return not none_float32_from_iter(query, db)

cpdef size_t count_if_float32(float32_t[:] query, Float32Set db) except *:
    """
        returns the number of (non-unique) elements in query, which are also in db
    """
    if query is None or db is None:
        return 0
    cdef size_t i
    cdef size_t n=len(query)
    cdef size_t res=0
    for i in range(n):
        if db.contains(query[i]):
            res+=1
    return res

cpdef size_t count_if_float32_from_iter(object query, Float32Set db) except *:
    """
        returns the number of (non-unique) elements in query (as iter), which are also in db
    """
    if query is None or db is None:
        return 0
    cdef float32_t el
    cdef size_t res=0
    for el in query:
        if db.contains(el):
            res+=1
    return res

cpdef bint aredisjoint_float32(Float32Set a, Float32Set b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef Float32SetIterator it
    cdef Float32Set s
    cdef float32_t el
    if a.size()<b.size():
        it=a.get_iter()
        s =b
    else:
        it=b.get_iter()
        s =a
    while it.has_next():
        el = it.next()
        if s.contains(el):
            return False
    return True

cpdef Float32Set intersect_float32(Float32Set a, Float32Set b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef Float32Set result = Float32Set()
    cdef Float32SetIterator it
    cdef Float32Set s
    cdef float32_t el
    if a.size()<b.size():
        it=a.get_iter()
        s =b
    else:
        it=b.get_iter()
        s =a
    while it.has_next():
        el = it.next()
        if s.contains(el):
            result.add(el)
    return result

cpdef bint issubset_float32(Float32Set s, Float32Set sub) except *:
    if s is None or sub is None:
        raise TypeError("'NoneType' object is not iterable")

    if s.size() < sub.size():
        return False

    cdef Float32SetIterator it=sub.get_iter()
    cdef float32_t el
    while it.has_next():
        el = it.next()
        if not s.contains(el):
            return False
    return True

cpdef Float32Set copy_float32(Float32Set s):
    if s is None:
        return None
    cdef Float32Set result = Float32Set(number_of_elements_hint=s.size())
    cdef Float32SetIterator it=s.get_iter()
    cdef float32_t el
    while it.has_next():
        el = it.next()
        result.add(el)
    return result

cpdef void update_float32(Float32Set s, Float32Set other) except *:
    if s is None or other is None:
        raise TypeError("'NoneType' object is not iterable")
    cdef Float32SetIterator it=other.get_iter()
    cdef float32_t el
    while it.has_next():
        el = it.next()
        s.add(el)

cpdef void swap_float32(Float32Set a, Float32Set b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef kh_float32set_t *tmp=a.table
    a.table=b.table
    b.table=tmp

cpdef Float32Set difference_float32(Float32Set a, Float32Set b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef float32_t el
    cdef Float32Set result = Float32Set()
    cdef Float32SetIterator it = a.get_iter()
    while it.has_next():
        el = it.next()
        if not b.contains(el):
            result.add(el)
    return result


cpdef Float32Set symmetric_difference_float32(Float32Set a, Float32Set b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef float32_t el
    cdef Float32Set result = Float32Set()
    cdef Float32SetIterator it = a.get_iter()
    while it.has_next():
          el = it.next()
          if not b.contains(el):
                result.add(el)
    it = b.get_iter()
    while it.has_next():
        el = it.next()
        if not a.contains(el):
            result.add(el)
    return result


#### special for PyObject:


cdef void _dealloc_pyobject(kh_pyobjectset_t *table) except *:
    cdef khint_t i = 0
    if table is not NULL:
        for i in range(table.size):
            if kh_exist_pyobjectset(table, i):
                Py_DECREF(<object>table.keys[i])
        kh_destroy_pyobjectset(table)

cdef bint _contains_pyobject(kh_pyobjectset_t *table, object key) nogil:
        cdef khint_t k
        k = kh_get_pyobjectset(table, <pyobject_t>key)
        return k != table.n_buckets

cdef void _add_pyobject(kh_pyobjectset_t *table, object key) except *:
        cdef:
            khint_t k
            int ret = 0
            pyobject_t key_ptr = <pyobject_t> key
        k = kh_put_pyobjectset(table, key_ptr, &ret)
        if ret: 
            #element was really added, so we need to increase reference
            Py_INCREF(key)

cdef void _discard_pyobject(kh_pyobjectset_t *table, object key) except *:
    cdef khint_t k
    cdef pyobject_t key_ptr = <pyobject_t> key
    k = kh_get_pyobjectset(table, key_ptr)
    if k != table.n_buckets:
        Py_DECREF(<object>table.keys[k])
        kh_del_pyobjectset(table, k)


### Iterator:
cdef class PyObjectSetIterator:

    cdef void __move(self) except *:
        while self.it<self.parent.table.n_buckets and not kh_exist_pyobjectset(self.parent.table, self.it):
              self.it+=1       

    cdef bint has_next(self) except *:
        self.__move()
        return self.it < self.parent.table.n_buckets
        

    # doesn't work if there was change between last has_next() and next() 
    cdef object next(self):
        cdef pyobject_t result = self.parent.table.keys[self.it]
        self.it+=1#ensure at least one move!
        return <object>result


    def __cinit__(self, PyObjectSet parent):
        self.parent = parent
        #search the start:
        self.it = 0
        self.__move()

    def __next__(self):
        if self.has_next():
            return self.next()
        else:
            raise StopIteration

#### the same for all:

cdef class PyObjectSet:

    def __cinit__(self, iterable=None, *, number_of_elements_hint=None):
        """
        iterable - initial elements in the set
        number_of_elements_hint - number of elements without the need of reallocation.
        """
        self.table = kh_init_pyobjectset()
        if number_of_elements_hint is not None:    
            kh_resize_pyobjectset(self.table, element_n_to_bucket_n(number_of_elements_hint))
        cdef object el
        if iterable is not None:
            for el in iterable:
                self.add(el)

    def __len__(self):
        return self.size()
  
    cdef khint_t size(self):
        return self.table.size
        

    def __dealloc__(self):
        _dealloc_pyobject(self.table)
        self.table = NULL

    def __contains__(self, object key):
        return self.contains(key)


    cdef bint contains(self, object key) except *:
        return _contains_pyobject(self.table, key)


    cpdef void add(self, object key) except *:
        _add_pyobject(self.table, key)

    
    cpdef void discard(self, object key) except *:
        _discard_pyobject(self.table, key)


    cdef PyObjectSetIterator get_iter(self):
        return PyObjectSetIterator(self)

    def __iter__(self):
        return self.get_iter()

    def get_state_info(self):
        """
        returns information about state of the set

        >>> from cykhash import PyObjectSet
        >>> info = PyObjectSet([1]).get_state_info()
        >>> info["n_buckets"]
        4
        >>> info["n_occupied"]
        1

        """
        return {"n_buckets" : self.table.n_buckets, 
                "n_occupied" : self.table.n_occupied, 
                "upper_bound" : self.table.upper_bound}

    ### drop-in for set:
    def isdisjoint(self, other):
        if isinstance(other, PyObjectSet):
            return aredisjoint_pyobject(self, other)
        cdef object el
        for el in other:
            if self.contains(el):
                return False
        return True

    def issuperset(self, other):
        if isinstance(other, PyObjectSet):
            return issubset_pyobject(self, other)
        cdef object el
        for el in other:
            if not self.contains(el):
                return False
        return True

    def issubset(self, other):
        if isinstance(other, PyObjectSet):
            return issubset_pyobject(other, self)
        cdef object el
        cdef PyObjectSet mem=PyObjectSet()
        for el in other:
            if self.contains(el):
                mem.add(el)
        return mem.size()==self.size()

    def __repr__(self):
        return "{"+','.join(map(str, self))+"}"

    def __le__(self, PyObjectSet other):
        return issubset_pyobject(other, self)

    def __lt__(self, PyObjectSet other):
        return issubset_pyobject(other, self) and self.size()<other.size()

    def __ge__(self, PyObjectSet other):
        return issubset_pyobject(self,  other)

    def __gt__(self, PyObjectSet other):
        return issubset_pyobject(self, other) and self.size()>other.size()

    def __eq__(self, PyObjectSet other):
        return issubset_pyobject(self, other) and self.size()==other.size()

    def __or__(self, PyObjectSet other):
        cdef PyObjectSet res = copy_pyobject(self)
        update_pyobject(res, other)
        return res

    def __ior__(self, PyObjectSet other):
        update_pyobject(self, other)
        return self

    def __and__(self, PyObjectSet other):
        return intersect_pyobject(self, other)

    def __iand__(self, PyObjectSet other):
        cdef PyObjectSet res = intersect_pyobject(self, other)
        swap_pyobject(self, res)
        return self

    def __sub__(self, PyObjectSet other):
        return difference_pyobject(self, other)

    def __isub__(self, PyObjectSet other):
        cdef PyObjectSet res = difference_pyobject(self, other)
        swap_pyobject(self, res)
        return self

    def __xor__(self, PyObjectSet other):
        return symmetric_difference_pyobject(self, other)

    def __ixor__(self, PyObjectSet other):
        cdef PyObjectSet res = symmetric_difference_pyobject(self, other)
        swap_pyobject(self, res)
        return self

    def copy(self):
        return copy_pyobject(self)

    def union(self, *others):
        cdef PyObjectSet res = copy_pyobject(self)
        for o in others:
            res.update(o)
        return res

    def update(self, other):
        if isinstance(other, PyObjectSet):
            update_pyobject(self, other)
            return
        cdef object el
        for el in other:
            self.add(el)

    def intersection(self, *others):
        cdef PyObjectSet res = copy_pyobject(self)
        for o in others:
            res.intersection_update(o)
        return res

    def intersection_update(self, other):
        cdef PyObjectSet res 
        cdef object el
        if isinstance(other, PyObjectSet):
            res = intersect_pyobject(self, other)
        else:
            res = PyObjectSet()
            for el in other:
                if self.contains(el):
                    res.add(el)
        swap_pyobject(self, res)

    def difference_update(self, other):
        cdef PyObjectSet res 
        cdef object el
        if isinstance(other, PyObjectSet):
            res = difference_pyobject(self, other)
            swap_pyobject(self, res)
        else:
            for el in other:
                self.discard(el)

    def difference(self, *others):
        cdef PyObjectSet res = copy_pyobject(self)
        for o in others:
            res.difference_update(o)
        return res

    def symmetric_difference_update(self, other):
        cdef PyObjectSet res 
        cdef object el
        if isinstance(other, PyObjectSet):
            res = symmetric_difference_pyobject(self, other)
        else:
            res = self.copy()
            for el in other:
                if self.contains(el):
                    res.discard(el)
                else:
                    res.add(el)
        swap_pyobject(self, res)

    def symmetric_difference(self, *others):
        cdef PyObjectSet res = copy_pyobject(self)
        for o in others:
            res.symmetric_difference_update(o)
        return res

    def clear(self):
        cdef PyObjectSet res = PyObjectSet()
        swap_pyobject(self, res)

    def remove(self, key):
        cdef size_t old=self.size()
        self.discard(key)
        if old==self.size():
            raise KeyError(key)

    def pop(self):
        if self.size()== 0:
            raise KeyError("pop from empty set")
        cdef PyObjectSetIterator it = self.get_iter()
        cdef object el = it.next()
        self.discard(el)
        return el
        



### Utils:

def PyObjectSet_from(it):
    """
        creates PyObjectSet from an iterator. 
        Use PyObjectSet_from_buffer for a faster version if iterator is buffer of correct type
    """
    res=PyObjectSet()
    for i in it:
        res.add(i)
    return res

cpdef PyObjectSet PyObjectSet_from_buffer(object[:] buf, double size_hint=0.0):
    """
        creates PyObjectSet from the given buffer buf. 
        Use slower PyObjectSet_from if series is given as iterator without buffer protocol.
        size_hint is an estimation of the ratio of unique elements. The default value of 0.0 means all elements in buf are expected to be unique
        Giving a good estimate will avoid rehashing (if estimate is too low) and having too big table (if estimate is too high).
    """
    cdef Py_ssize_t n = len(buf)
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=PyObjectSet(number_of_elements_hint=at_least_needed)
    cdef Py_ssize_t i
    for i in range(n):
        res.add(buf[i])
    return res
    

cpdef void isin_pyobject(object[:] query, PyObjectSet db, uint8_t[:] result) except *:
    """
        given query, db writes for every element of query True/False into result depending on whether query-element is in db (=True) or not (=False). 
        result should have the same length as query.
    """
    cdef size_t i
    cdef size_t n=len(query)
    if n!=len(result):
        raise ValueError("Different sizes for query({n}) and result({m})".format(n=n, m=len(result)))
    for i in range(n):
        result[i]=db is not None and db.contains(query[i])

cpdef bint all_pyobject(object[:] query, PyObjectSet db) except *:
    """
        True if all elements of query are in db, False otherwise.
    """
    if query is None:
        return True
    cdef size_t i
    cdef size_t n=len(query)
    if db is None:
        return n==0
    for i in range(n):
        if not db.contains(query[i]):
            return False
    return True

cpdef bint all_pyobject_from_iter(object query, PyObjectSet db) except *:
    """
        True if all elements of query (as iterator) are in db, False otherwise.
    """
    if query is None:
        return True
    cdef object el
    for el in query:
        if db is None or not db.contains(el):
            return False
    return True

cpdef bint none_pyobject(object[:] query, PyObjectSet db) except *:
    """
        True if none of elements in query is in db, False otherwise.
    """
    if query is None or db is None:
        return True
    cdef size_t i
    cdef size_t n=len(query)
    for i in range(n):
        if db.contains(query[i]):
            return False
    return True

cpdef bint none_pyobject_from_iter(object query, PyObjectSet db) except *:
    """
        True if none of elements in query (as iterator) is in db, False otherwise.
    """
    if query is None or db is None:
        return True
    cdef object el
    for el in query:
        if db.contains(el):
            return False
    return True

cpdef bint any_pyobject(object[:] query, PyObjectSet db) except *:
    """
        True if one of elements in query is in db, False otherwise.
    """
    return not none_pyobject(query, db)

cpdef bint any_pyobject_from_iter(object query, PyObjectSet db) except *:
    """
        True if one of elements in query (as iterator) is in db, False otherwise.
    """
    return not none_pyobject_from_iter(query, db)

cpdef size_t count_if_pyobject(object[:] query, PyObjectSet db) except *:
    """
        returns the number of (non-unique) elements in query, which are also in db
    """
    if query is None or db is None:
        return 0
    cdef size_t i
    cdef size_t n=len(query)
    cdef size_t res=0
    for i in range(n):
        if db.contains(query[i]):
            res+=1
    return res

cpdef size_t count_if_pyobject_from_iter(object query, PyObjectSet db) except *:
    """
        returns the number of (non-unique) elements in query (as iter), which are also in db
    """
    if query is None or db is None:
        return 0
    cdef object el
    cdef size_t res=0
    for el in query:
        if db.contains(el):
            res+=1
    return res

cpdef bint aredisjoint_pyobject(PyObjectSet a, PyObjectSet b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef PyObjectSetIterator it
    cdef PyObjectSet s
    cdef object el
    if a.size()<b.size():
        it=a.get_iter()
        s =b
    else:
        it=b.get_iter()
        s =a
    while it.has_next():
        el = it.next()
        if s.contains(el):
            return False
    return True

cpdef PyObjectSet intersect_pyobject(PyObjectSet a, PyObjectSet b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef PyObjectSet result = PyObjectSet()
    cdef PyObjectSetIterator it
    cdef PyObjectSet s
    cdef object el
    if a.size()<b.size():
        it=a.get_iter()
        s =b
    else:
        it=b.get_iter()
        s =a
    while it.has_next():
        el = it.next()
        if s.contains(el):
            result.add(el)
    return result

cpdef bint issubset_pyobject(PyObjectSet s, PyObjectSet sub) except *:
    if s is None or sub is None:
        raise TypeError("'NoneType' object is not iterable")

    if s.size() < sub.size():
        return False

    cdef PyObjectSetIterator it=sub.get_iter()
    cdef object el
    while it.has_next():
        el = it.next()
        if not s.contains(el):
            return False
    return True

cpdef PyObjectSet copy_pyobject(PyObjectSet s):
    if s is None:
        return None
    cdef PyObjectSet result = PyObjectSet(number_of_elements_hint=s.size())
    cdef PyObjectSetIterator it=s.get_iter()
    cdef object el
    while it.has_next():
        el = it.next()
        result.add(el)
    return result

cpdef void update_pyobject(PyObjectSet s, PyObjectSet other) except *:
    if s is None or other is None:
        raise TypeError("'NoneType' object is not iterable")
    cdef PyObjectSetIterator it=other.get_iter()
    cdef object el
    while it.has_next():
        el = it.next()
        s.add(el)

cpdef void swap_pyobject(PyObjectSet a, PyObjectSet b) except *:
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef kh_pyobjectset_t *tmp=a.table
    a.table=b.table
    b.table=tmp

cpdef PyObjectSet difference_pyobject(PyObjectSet a, PyObjectSet b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef object el
    cdef PyObjectSet result = PyObjectSet()
    cdef PyObjectSetIterator it = a.get_iter()
    while it.has_next():
        el = it.next()
        if not b.contains(el):
            result.add(el)
    return result


cpdef PyObjectSet symmetric_difference_pyobject(PyObjectSet a, PyObjectSet b):
    if a is None or b is None:
        raise TypeError("'NoneType' object is not iterable")

    cdef object el
    cdef PyObjectSet result = PyObjectSet()
    cdef PyObjectSetIterator it = a.get_iter()
    while it.has_next():
          el = it.next()
          if not b.contains(el):
                result.add(el)
    it = b.get_iter()
    while it.has_next():
        el = it.next()
        if not a.contains(el):
            result.add(el)
    return result

