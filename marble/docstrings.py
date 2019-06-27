import pprint

__all__ = ['document_properties',]


def format_property_dict(properties, indent=0):
    return_string = ''
    prefix = ' ' * indent
    for name, data in properties.items():
        return_string += prefix + '{}:\n'.format(name)
        for prop in ('alias', 'dims', 'units'):
            if prop in data:
                return_string += prefix + '    {}: {},\n'.format(prop, data[prop])
    return return_string


def _document_property(obj, property_name):
    if hasattr(obj, property_name):
        return_string = '\n\n{}:\n\n'.format(property_name.replace('_', ' ').title())
        return_string += format_property_dict(getattr(obj, property_name), indent=4)
    else:
        return_string = ''
    return return_string


def document_properties(obj):
    docstring = obj.__doc__ or ''
    for property_name in ('input_properties', 'diagnostic_properties', 'tendency_properties', 'output_properties'):
        docstring += _document_property(obj, property_name)
    obj.__doc__ = docstring
    if docstring:
        obj.__doc__ += '\n'
    return obj
