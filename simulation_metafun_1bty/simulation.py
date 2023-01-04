import sys
import os
import os.path as osp
import pickle
import random

import numpy as np
import pickle as pkl

import openmm.app as omma
import openmm as omm
from openmmplumed import PlumedForce
import simtk.unit as unit


import mdtraj as mdj
import time

# from wepy, to restart a simulation
GET_STATE_KWARG_DEFAULTS = (('getPositions', True),
                            ('getVelocities', True),
                            ('getForces', True),
                            ('getEnergy', True),
                            ('getParameters', True),
                            ('getParameterDerivatives', True),
                            ('enforcePeriodicBox', True),)
# Platform used for OpenMM which uses different hardware computation
# kernels. Options are: Reference, CPU, OpenCL, CUDA.

PLATFORM = 'CUDA'
PRECISION = 'mixed'
TEMPERATURE = 300.0 * unit.kelvin
FRICTION_COEFFICIENT = 1.0 / unit.picosecond
STEP_SIZE = 0.002 * unit.picoseconds
PRESSURE = 1.0 * unit.atmosphere
VOLUME_MOVE_FREQ = 50

# reporter
NUM_STEPS = 500000000 #1micros, 500000 = 1ns
DCD_REPORT_STEPS = 500
CHECKPOINT_REPORTER_STEPS = 10000
OUTPUTS_PATH = osp.realpath(f'outputs')
SIM_TRAJ = 'traj.dcd'
CHECKPOINTLAST = 'checkpint_last.chk'
SYSTEM_FILE = 'system.pkl'
OMM_STATE_FILE = 'state.pkl'
CHECKPOINT = 'checkpoint.chk'

#
if not osp.exists(OUTPUTS_PATH):
    os.makedirs(OUTPUTS_PATH)

# the inputs directory and files we need
inputs_dir = osp.realpath(f'./inputs/')
prmtop = omma.amberprmtopfile.AmberPrmtopFile("inputs/complex.prmtop")
inpcrd = omma.amberinpcrdfile.AmberInpcrdFile("inputs/complex.rst7")
plumed_file = osp.join(inputs_dir, 'plumed.dat')

# select protein and ligand indices
pdb = mdj.load_pdb(osp.join(inputs_dir, "complex.pdb"))
protein_ligand_idxs = pdb.topology.select('protein or resname "MOL"')
#load gro and top files required

# add disulfide bonds to the topology
prmtop.topology.createDisulfideBonds(inpcrd.positions)

# build the system
system = prmtop.createSystem(nonbondedMethod=omma.PME, nonbondedCutoff=1*unit.nanometer,
                          constraints=omma.HBonds)

# # atm, 300 K, with volume move attempts every 50 steps
barostat = omm.MonteCarloBarostat(PRESSURE, TEMPERATURE, VOLUME_MOVE_FREQ)
# # add it as a "Force" to the system
system.addForce(barostat)

# add Plumed
with open(plumed_file, 'r') as file:
    script = file.read()
system.addForce(PlumedForce(script))

# make the integrator
integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

platform = omm.Platform.getPlatformByName(PLATFORM)
prop = dict(Precision=PRECISION)

simulation = omma.Simulation(prmtop.topology, system, integrator, platform, prop)
simulation.context.setPositions(inpcrd.positions)

simulation.reporters.append(mdj.reporters.DCDReporter(osp.join(OUTPUTS_PATH, SIM_TRAJ),
                                                               DCD_REPORT_STEPS,
                                                             atomSubset=protein_ligand_idxs))

simulation.reporters.append(omma.CheckpointReporter(osp.join(OUTPUTS_PATH, CHECKPOINT),
                                                    CHECKPOINT_REPORTER_STEPS))
#simulation.minimizeEnergy()
start_time = time.time()
simulation.step(NUM_STEPS)
end_time = time.time()
print(f"Run time = {np.round(end_time - start_time, 3)}s")
simulation_time = round((STEP_SIZE * NUM_STEPS).value_in_unit(unit.nanoseconds),
                           2)
print(f"Simulation time: {simulation_time}ns")
simulation.saveCheckpoint(osp.join(OUTPUTS_PATH, CHECKPOINTLAST))

# save final state and system
get_state_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
omm_state = simulation.context.getState(**get_state_kwargs)
# save the pkl files to the inputs dir
with open(osp.join(OUTPUTS_PATH, SYSTEM_FILE), 'wb') as wfile:
    pkl.dump(system, wfile)

with open(osp.join(OUTPUTS_PATH, OMM_STATE_FILE), 'wb') as wfile:
    pkl.dump(omm_state, wfile)
print('Done making pkls. Check inputs dir for them!')