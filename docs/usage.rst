=====
Usage
=====

To use MARBLE in a project::

    import marble

MARBLE uses the Sympl framework. You can read more in the `Sympl documentation`_.
MARBLE comes with code examples, which can be accessed from your local installation or the `MARBLE github repo`_.

Aliases
=======

As of writing this documentation, one shortcoming of Sympl is the need to explicitly
write the long name of any quantities that are accessed from a state dictionary
in the run script or analysis code. This package adds some helper tools that
avoid this requirement, by allowing you to refer to long names using *aliases*
that you register at the top of your run script. This retains the benefit of
having quantities explicitly defined, because anyone reading your code can look
at the top where you register aliases to figure out what your short aliases mean.

For example::

    import marble
    import sympl

    marble.register_alias('rt', 'total_water_mixing_ratio')
    # or
    # marble.register_alias_dict({'rt': 'total_water_mixing_ratio'})

    state = {
        'total_water_mixing_ratio': sympl.DataArray(0., dims=[], attrs={'units': 'kg/kg'})
    }

    state = marble.AliasDict(state)
    print(state['rt'])  # gets state['total_water_mixing_ratio']

.. autofunction:: marble.register_alias
.. autofunction:: marble.register_alias_dict
.. autoclass:: marble.AliasDict

Initialization
==============

State initialization is not performed by the MARBLE module, but we do give an
`example initialization code`_. To call the modules, you need to create a state
that has all the required quantities with defined dimensions and units.

One thing to keep in mind, which we will discuss more below, is that MARBLE runs
using principal components of its vertically-resolved quantities. Those
principal components are pre-defined, and assume that their height-resolved
inputs are on a 3km, 20-point equidistant grid with points at 0km and 3km.

Decomposition
=============

As we just said, MARBLE runs using principal components of its
vertically-resolved quantities. Those
principal components are pre-defined, and assume that their height-resolved
inputs are on a 3km, 20-point equidistant grid with points at 0km and 3km.

When MARBLE is run, it operates on principal components of vertically-resolved
quantities. This means that before integration, the state needs to
be converted into principal components, and after integration they need to be
converted back to height coordinates before plotting or analysis.

To convert between height and principal components, we provide two helper
functions that operate on one quantity at a time, and three Sympl components
which operate on commonly-grouped quantities.

.. autofunction:: marble.convert_height_to_principal_components
.. autofunction:: marble.convert_principal_components_to_height
.. autoclass:: marble.InputHeightToPrincipalComponents
.. autoclass:: marble.InputPrincipalComponentsToHeight
.. autoclass:: marble.DiagnosticPrincipalComponentsToHeight


Forcing
=======

We use an extremely simple component to apply horizontal advective forcings
that are defined in the state as tendencies to the prognostic quantities. The
horizontal advective forcings need to be defined in principal component space.
This can be achieved using :func:`marble.convert_height_to_principal_components`.

.. autoclass:: marble.LatentHorizontalAdvectiveForcing(TendencyComponent)


MARBLE
======

MARBLE itself is contained in a `TendencyComponent`. Note that the surface
latent and sensible heat fluxes should be expressed as downward values, as in
the flux into the surface.

.. autoclass:: marble.LatentMarble


.. _Sympl documentation: https://sympl.readthedocs.io/en/latest/
.. _MARBLE github repo: https://github.com/mcgibbon/marble/tree/master/examples
.. _example initialization code: https://github.com/mcgibbon/marble/blob/master/examples/initialization.py
