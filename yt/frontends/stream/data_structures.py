"""
Data structures for Streaming, in-memory datasets

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: Columbia University
Homepage: http://yt-project.org/
License:
  Copyright (C) 2011 Matthew Turk.  All Rights Reserved.

  This file is part of yt.

  yt is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import weakref
import numpy as np

from yt.utilities.io_handler import io_registry
from yt.funcs import *
from yt.config import ytcfg
from yt.data_objects.grid_patch import \
    AMRGridPatch
from yt.geometry.grid_geometry_handler import \
    GridGeometryHandler
from yt.data_objects.static_output import \
    StaticOutput
from yt.utilities.logger import ytLogger as mylog
from yt.data_objects.field_info_container import \
    FieldInfoContainer, NullFunc
from yt.utilities.lib import \
    get_box_grids_level
from yt.utilities.decompose import \
    decompose_array, get_psize
from yt.utilities.definitions import \
    mpc_conversion, sec_conversion

from .fields import \
    StreamFieldInfo, \
    add_stream_field, \
    KnownStreamFields

class StreamGrid(AMRGridPatch):
    """
    Class representing a single In-memory Grid instance.
    """

    __slots__ = ['proc_num']
    _id_offset = 0
    def __init__(self, id, hierarchy):
        """
        Returns an instance of StreamGrid with *id*, associated with *filename*
        and *hierarchy*.
        """
        #All of the field parameters will be passed to us as needed.
        AMRGridPatch.__init__(self, id, filename = None, hierarchy = hierarchy)
        self._children_ids = []
        self._parent_id = -1
        self.Level = -1

    def _guess_properties_from_parent(self):
        rf = self.pf.refine_by
        my_ind = self.id - self._id_offset
        le = self.LeftEdge
        self.dds = self.Parent.dds/rf
        ParentLeftIndex = np.rint((self.LeftEdge-self.Parent.LeftEdge)/self.Parent.dds)
        self.start_index = rf*(ParentLeftIndex + self.Parent.get_global_startindex()).astype('int64')
        self.LeftEdge = self.Parent.LeftEdge + self.Parent.dds * ParentLeftIndex
        self.RightEdge = self.LeftEdge + self.ActiveDimensions*self.dds
        self.hierarchy.grid_left_edge[my_ind,:] = self.LeftEdge
        self.hierarchy.grid_right_edge[my_ind,:] = self.RightEdge
        self._child_mask = None
        self._child_index_mask = None
        self._child_indices = None
        self._setup_dx()

    def set_filename(self, filename):
        pass

    def __repr__(self):
        return "StreamGrid_%04i" % (self.id)

    @property
    def Parent(self):
        if self._parent_id == -1: return None
        return self.hierarchy.grids[self._parent_id - self._id_offset]

    @property
    def Children(self):
        return [self.hierarchy.grids[cid - self._id_offset]
                for cid in self._children_ids]

class StreamHandler(object):
    def __init__(self, left_edges, right_edges, dimensions,
                 levels, parent_ids, particle_count, processor_ids,
                 fields, io = None, storage_filename = None):
        self.left_edges = left_edges
        self.right_edges = right_edges
        self.dimensions = dimensions
        self.levels = levels
        self.parent_ids = parent_ids
        self.particle_count = particle_count
        self.processor_ids = processor_ids
        self.num_grids = self.levels.size
        self.fields = fields
        self.io = io
        self.storage_filename = storage_filename

    def get_fields(self):
        return self.fields.all_fields

class StreamHierarchy(GridGeometryHandler):

    grid = StreamGrid

    def __init__(self, pf, data_style = None):
        self.data_style = data_style
        self.float_type = 'float64'
        self.parameter_file = weakref.proxy(pf) # for _obtain_enzo
        self.stream_handler = pf.stream_handler
        self.float_type = "float64"
        self.directory = os.getcwd()
        GridGeometryHandler.__init__(self, pf, data_style)

    def _count_grids(self):
        self.num_grids = self.stream_handler.num_grids

    def _parse_hierarchy(self):
        self.grid_dimensions = self.stream_handler.dimensions
        self.grid_left_edge[:] = self.stream_handler.left_edges
        self.grid_right_edge[:] = self.stream_handler.right_edges
        self.grid_levels[:] = self.stream_handler.levels
        self.grid_procs = self.stream_handler.processor_ids
        self.grid_particle_count[:] = self.stream_handler.particle_count
        mylog.debug("Copying reverse tree")
        self.grids = []
        # We enumerate, so it's 0-indexed id and 1-indexed pid
        for id in xrange(self.num_grids):
            self.grids.append(self.grid(id, self))
            self.grids[id].Level = self.grid_levels[id, 0]
        parent_ids = self.stream_handler.parent_ids
        if parent_ids is not None:
            reverse_tree = self.stream_handler.parent_ids.tolist()
            # Initial setup:
            for gid,pid in enumerate(reverse_tree):
                if pid >= 0:
                    self.grids[id]._parent_id = pid
                    self.grids[pid]._children_ids.append(self.grids[gid].id)
        else:
            mylog.debug("Reconstructing parent-child relationships")
            self._reconstruct_parent_child()
        self.max_level = self.grid_levels.max()
        mylog.debug("Preparing grids")
        temp_grids = np.empty(self.num_grids, dtype='object')
        for i, grid in enumerate(self.grids):
            if (i%1e4) == 0: mylog.debug("Prepared % 7i / % 7i grids", i, self.num_grids)
            grid.filename = None
            grid._prepare_grid()
            grid.proc_num = self.grid_procs[i]
            temp_grids[i] = grid
        self.grids = temp_grids
        mylog.debug("Prepared")

    def _reconstruct_parent_child(self):
        mask = np.empty(len(self.grids), dtype='int32')
        mylog.debug("First pass; identifying child grids")
        for i, grid in enumerate(self.grids):
            get_box_grids_level(self.grid_left_edge[i,:],
                                self.grid_right_edge[i,:],
                                self.grid_levels[i] + 1,
                                self.grid_left_edge, self.grid_right_edge,
                                self.grid_levels, mask)
            ids = np.where(mask.astype("bool"))
            grid._children_ids = ids[0] # where is a tuple
        mylog.debug("Second pass; identifying parents")
        for i, grid in enumerate(self.grids): # Second pass
            for child in grid.Children:
                child._parent_id = i

    def _initialize_grid_arrays(self):
        GridGeometryHandler._initialize_grid_arrays(self)
        self.grid_procs = np.zeros((self.num_grids,1),'int32')

    def _setup_classes(self):
        dd = self._get_data_reader_dict()
        GridGeometryHandler._setup_classes(self, dd)

    def _detect_fields(self):
        self.field_list = list(set(self.stream_handler.get_fields()))

    def _populate_grid_objects(self):
        for g in self.grids:
            g._setup_dx()
        self.max_level = self.grid_levels.max()

    def _setup_data_io(self):
        if self.stream_handler.io is not None:
            self.io = self.stream_handler.io
        else:
            self.io = io_registry[self.data_style](self.stream_handler)

class StreamStaticOutput(StaticOutput):
    _hierarchy_class = StreamHierarchy
    _fieldinfo_fallback = StreamFieldInfo
    _fieldinfo_known = KnownStreamFields
    _data_style = 'stream'

    def __init__(self, stream_handler):
        #if parameter_override is None: parameter_override = {}
        #self._parameter_override = parameter_override
        #if conversion_override is None: conversion_override = {}
        #self._conversion_override = conversion_override

        self.stream_handler = stream_handler
        StaticOutput.__init__(self, "InMemoryParameterFile", self._data_style)

        self.units = {}
        self.time_units = {}

    def _parse_parameter_file(self):
        self.basename = self.stream_handler.name
        self.parameters['CurrentTimeIdentifier'] = time.time()
        self.unique_identifier = self.parameters["CurrentTimeIdentifier"]
        self.domain_left_edge = self.stream_handler.domain_left_edge[:]
        self.domain_right_edge = self.stream_handler.domain_right_edge[:]
        self.refine_by = self.stream_handler.refine_by
        self.dimensionality = self.stream_handler.dimensionality
        self.domain_dimensions = self.stream_handler.domain_dimensions
        self.current_time = self.stream_handler.simulation_time
        if self.stream_handler.cosmology_simulation:
            self.cosmological_simulation = 1
            self.current_redshift = self.stream_handler.current_redshift
            self.omega_lambda = self.stream_handler.omega_lambda
            self.omega_matter = self.stream_handler.omega_matter
            self.hubble_constant = self.stream_handler.hubble_constant
        else:
            self.current_redshift = self.omega_lambda = self.omega_matter = \
                self.hubble_constant = self.cosmological_simulation = 0.0

    def _set_units(self):
        pass

    @classmethod
    def _is_valid(cls, *args, **kwargs):
        return False

class StreamDictFieldHandler(dict):

    @property
    def all_fields(self): return self[0].keys()

def load_uniform_grid(data, domain_dimensions, sim_unit_to_cm, bbox=None,
                      nprocs=1, sim_time=0.0, number_of_particles=0):
    r"""Load a uniform grid of data into yt as a
    :class:`~yt.frontends.stream.data_structures.StreamHandler`.

    This should allow a uniform grid of data to be loaded directly into yt and
    analyzed as would any others.  This comes with several caveats:
        * Units will be incorrect unless the data has already been converted to
          cgs.
        * Some functions may behave oddly, and parallelism will be
          disappointing or non-existent in most cases.
        * Particles may be difficult to integrate.

    Parameters
    ----------
    data : dict
        This is a dict of numpy arrays, where the keys are the field names.
    domain_dimensions : array_like
        This is the domain dimensions of the grid
    sim_unit_to_cm : float
        Conversion factor from simulation units to centimeters
    bbox : array_like (xdim:zdim, LE:RE), optional
        Size of computational domain in units sim_unit_to_cm
    nprocs: integer, optional
        If greater than 1, will create this number of subarrays out of data
    sim_time : float, optional
        The simulation time in seconds
    number_of_particles : int, optional
        If particle fields are included, set this to the number of particles

    Examples
    --------

    >>> arr = np.random.random((128, 128, 129))
    >>> data = dict(Density = arr)
    >>> bbox = np.array([[0., 1.0], [-1.5, 1.5], [1.0, 2.5]])
    >>> pf = load_uniform_grid(data, arr.shape, 3.08e24, bbox=bbox, nprocs=12)

    """

    domain_dimensions = np.array(domain_dimensions)
    if bbox is None:
        bbox = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], 'float64')
    domain_left_edge = np.array(bbox[:, 0], 'float64')
    domain_right_edge = np.array(bbox[:, 1], 'float64')
    grid_levels = np.zeros(nprocs, dtype='int32').reshape((nprocs,1))

    sfh = StreamDictFieldHandler()

    if nprocs > 1:
        temp = {}
        new_data = {}
        for key in data.keys():
            psize = get_psize(np.array(data[key].shape), nprocs)
            grid_left_edges, grid_right_edges, temp[key] = \
                decompose_array(data[key], psize, bbox)
            grid_dimensions = np.array([grid.shape for grid in temp[key]],
                                       dtype="int32")
        for gid in range(nprocs):
            new_data[gid] = {}
            for key in temp.keys():
                new_data[gid].update({key:temp[key][gid]})
        sfh.update(new_data)
        del new_data, temp
    else:
        sfh.update({0:data})
        grid_left_edges = domain_left_edge
        grid_right_edges = domain_right_edge
        grid_dimensions = domain_dimensions.reshape(nprocs,3).astype("int32")

    handler = StreamHandler(
        grid_left_edges,
        grid_right_edges,
        grid_dimensions,
        grid_levels,
        -np.ones(nprocs, dtype='int64'),
        number_of_particles*np.ones(nprocs, dtype='int64').reshape(nprocs,1),
        np.zeros(nprocs).reshape((nprocs,1)),
        sfh,
    )

    handler.name = "UniformGridData"
    handler.domain_left_edge = domain_left_edge
    handler.domain_right_edge = domain_right_edge
    handler.refine_by = 2
    handler.dimensionality = 3
    handler.domain_dimensions = domain_dimensions
    handler.simulation_time = sim_time
    handler.cosmology_simulation = 0

    spf = StreamStaticOutput(handler)
    spf.units["cm"] = sim_unit_to_cm
    spf.units['1'] = 1.0
    spf.units["unitary"] = 1.0
    box_in_mpc = sim_unit_to_cm / mpc_conversion['cm']
    for unit in mpc_conversion.keys():
        spf.units[unit] = mpc_conversion[unit] * box_in_mpc
    return spf
