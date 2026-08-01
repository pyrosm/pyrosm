



# different implementations:
include "maps/map_impl.pxi"



# backward compartibility:
def Float64to64Map(*, number_of_elements_hint=None, for_int=True):
    if for_int:
        return Float64toInt64Map(number_of_elements_hint=number_of_elements_hint)
    else:
        return Float64toFloat64Map(number_of_elements_hint=number_of_elements_hint)


def Int64to64Map(*, number_of_elements_hint=None, for_int=True):
    if for_int:
        return Int64toInt64Map(number_of_elements_hint=number_of_elements_hint)
    else:
        return Int64toFloat64Map(number_of_elements_hint=number_of_elements_hint)


def Float32to32Map(*, number_of_elements_hint=None, for_int=True):
    if for_int:
        return Float32toInt32Map(number_of_elements_hint=number_of_elements_hint)
    else:
        return Float32toFloat32Map(number_of_elements_hint=number_of_elements_hint)


def Int32to32Map(*, number_of_elements_hint=None, for_int=True):
    if for_int:
        return Int32toInt32Map(number_of_elements_hint=number_of_elements_hint)
    else:
        return Int32toFloat32Map(number_of_elements_hint=number_of_elements_hint)
