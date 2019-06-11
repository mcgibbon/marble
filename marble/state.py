
_alias_to_long_name = {}


def register_alias(alias, long_name):
    _alias_to_long_name[alias] = long_name


def register_alias_dict(alias_dict):
    for alias, long_name in alias_dict.items():
        register_alias(alias, long_name)


class AliasDict(dict):

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        super(AliasDict, self).update(*args, **kwargs)

    def __getitem__(self, key):
        if key not in self and key in _alias_to_long_name:
            val = dict.__getitem__(self, _alias_to_long_name[key])
        else:
            val = dict.__getitem__(self, key)
        return val

    def __setitem__(self, key, val):
        if key in _alias_to_long_name:
            key = _alias_to_long_name[key]
        dict.__setitem__(self, key, val)

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '%s(%s)' % (type(self).__name__, dictrepr)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).iteritems():
            self[k] = v
