from openmm.app import *
from openmm import *
from openmm.unit import *
from metadynamics import *
import mdtraj as mdj
from glob import glob
import os
import shutil

# System parameters.
T = 300.0 * kelvin
P = 1.01325 * bar
steps = 500000000 #1micros, 500000 = 1ns
hill_frequency = 1000
dt = 0.002  # Timestep (ps)
PRECISION = 'mixed'

# Load the topology and coordinate files.
prmtop = AmberPrmtopFile("inputs/complex.prmtop")
inpcrd = AmberInpcrdFile("inputs/complex.rst7")

# select protein and ligand indices
pdb = mdj.load_pdb('inputs/complex.pdb')
protein_ligand_idxs = pdb.topology.select('protein or resname "MOL"')
lig = pdb.topology.select('resname "MOL"')


# Initialise the molecular system.
system = prmtop.createSystem(
    nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds
)

# Add a barostat to run at constant pressure.
barostat = MonteCarloBarostat(P, T)
system.addForce(barostat)

# # Set funnel variables.
p1 = [4, 327, 347, 357, 529, 549, 566, 1168, 1737, 1744, 1768, 1775, 2466, 2478, 2489, 2499, 2516, 2523, 2535,
  2546, 2553, 2718, 2734, 2745, 2769, 2776, 2787, 2794, 2897, 2903, 2910, 2926]
p2 = [347, 549, 1768, 1775, 2478, 2489, 2499, 2516, 2523, 2535, 2546, 2718, 2734, 2745, 2769, 2776, 2787, 2794, 2903, 2910]


# # Create the bias variable for the funnel projection.
projection = CustomCentroidBondForce(3, "distance(g1,g2)*cos(angle(g1,g2,g3))")
projection.addGroup(lig)
projection.addGroup(p1)
projection.addGroup(p2)
projection.addBond([0, 1, 2])
projection.setUsesPeriodicBoundaryConditions(True)
sigma_proj = 0.025
proj = BiasVariable(projection, 0.3, 3.7, sigma_proj, False, gridWidth=200)

# Create the bias variable for the funnel extent.`
extent = CustomCentroidBondForce(3, "distance(g1,g2)*sin(angle(g1,g2,g3))")
extent.addGroup(lig)
extent.addGroup(p1)
extent.addGroup(p2)
extent.addBond([0, 1, 2])
extent.setUsesPeriodicBoundaryConditions(True)
sigma_ext = 0.05
ext = BiasVariable(extent, 0.0, 0.95, sigma_ext, False, gridWidth=200)

# # Add restraints.
k1 = 10000 * kilojoules_per_mole
k2 = 1000 * kilojoules_per_mole
lower_wall = 0.5 * nanometer
upper_wall = 3.5 * nanometer

# Upper wall.
upper_wall_rest = CustomCentroidBondForce(
    3, "(k/2)*max(distance(g1,g2)*cos(angle(g1,g2,g3)) - upper_wall, 0)^2"
)
upper_wall_rest.addGroup(lig)
upper_wall_rest.addGroup(p1)
upper_wall_rest.addGroup(p2)
upper_wall_rest.addBond([0, 1, 2])
upper_wall_rest.addGlobalParameter("k", k1)
upper_wall_rest.addGlobalParameter("upper_wall", upper_wall)
upper_wall_rest.setUsesPeriodicBoundaryConditions(True)
system.addForce(upper_wall_rest)

# Sides of the funnel.
wall_width = 0.85 * nanometer
wall_buffer = 0.15 * nanometer
beta_cent = 1.0
s_cent = 2.3 * nanometer
dist_restraint = CustomCentroidBondForce(
    3,
    "(k/2)*max(distance(g1,g2)*sin(angle(g1,g2,g3)) - (a/(1+exp(b*(distance(g1,g2)*cos(angle(g1,g2,g3))-c)))+d), 0)^2",
)
dist_restraint.addGroup(lig)
dist_restraint.addGroup(p1)
dist_restraint.addGroup(p2)
dist_restraint.addBond([0, 1, 2])
dist_restraint.addGlobalParameter("k", k2)
dist_restraint.addGlobalParameter("a", wall_width)
dist_restraint.addGlobalParameter("b", beta_cent)
dist_restraint.addGlobalParameter("c", s_cent)
dist_restraint.addGlobalParameter("d", wall_buffer)
dist_restraint.setUsesPeriodicBoundaryConditions(True)
system.addForce(dist_restraint)

# Lower wall.
lower_wall_rest = CustomCentroidBondForce(
    3, "(k/2)*min(distance(g1,g2)*cos(angle(g1,g2,g3)) - lower_wall, 0)^2"
)
lower_wall_rest.addGroup(lig)
lower_wall_rest.addGroup(p1)
lower_wall_rest.addGroup(p2)
lower_wall_rest.addBond([0, 1, 2])
lower_wall_rest.addGlobalParameter("k", k1)
lower_wall_rest.addGlobalParameter("lower_wall", lower_wall)
lower_wall_rest.setUsesPeriodicBoundaryConditions(True)
system.addForce(lower_wall_rest)

# Initialise the metadynamics object.
bias = 10.0
meta = Metadynamics(
    system,
    [proj, ext],
    T,
    bias,
    1.5 * kilojoules_per_mole,
    hill_frequency,
    biasDir=".",
    saveFrequency=hill_frequency,
)

# Define the integrator.
integrator = LangevinIntegrator(T, 1 / picosecond, dt * picoseconds)

# Set the simulation platform.
platform = Platform.getPlatformByName("CUDA")
properties = dict(Precision=PRECISION)


# Initialise and configure the simulation object.
simulation = Simulation(prmtop.topology, system, integrator, platform, properties)
simulation.context.setPositions(inpcrd.positions)

# Setting intial system velocities.
simulation.context.setVelocitiesToTemperature(T)

# Look for a restart file.
if os.path.isfile("outputs/simulation.chk"):
    simulation.loadCheckpoint("outputs/simulation.chk")
    shutil.copy("simulation.log", "old_simulation.log")
    sim_log_file = [line[:-2] for line in open("simulation.log").readlines()]
    current_steps = int(sim_log_file[-1].split(" ")[1])
    steps -= current_steps
    shutil.copy("COLVAR.npy", "old_COLVAR.npy")
    shutil.copy("HILLS", "old_HILLS")
    shutil.copy("simulation.dcd", "old_simulation.dcd")

# Add reporters.
simulation.reporters.append(mdj.reporters.DCDReporter("outputs/simulation.dcd", 10000, atomSubset=protein_ligand_idxs))
simulation.reporters.append(
    StateDataReporter(
        "simulation.log",
        100000,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        volume=True,
        temperature=True,
        totalSteps=True,
        separator=" ",
    )
)
simulation.reporters.append(CheckpointReporter("outputs/simulation.chk", 10000))

# Create PLUMED compatible HILLS file.
file = open("HILLS", "w")
file.write("#! FIELDS time pp.proj pp.ext sigma_pp.proj sigma_pp.ext height biasf\n")
file.write("#! SET multivariate false\n")
file.write("#! SET kerneltype gaussian\n")

# Initialise the collective variable array.
current_cvs = np.array(
    list(meta.getCollectiveVariables(simulation)) + [meta.getHillHeight(simulation)]
)
colvar_array = np.array([current_cvs])

# Write the inital collective variable record.
line = colvar_array[0]
time = 0
write_line = f"{time:15} {line[0]:20.16f} {line[1]:20.16f}          {sigma_proj}           {sigma_ext} {line[2]:20.16f}            {bias}\n"
file.write(write_line)

# Run the simulation.
cycles = steps // hill_frequency
for x in range(0, cycles):
    meta.step(simulation, hill_frequency)
    current_cvs = np.array(
        list(meta.getCollectiveVariables(simulation)) + [meta.getHillHeight(simulation)]
    )
    colvar_array = np.append(colvar_array, [current_cvs], axis=0)
    np.save("COLVAR.npy", colvar_array)
    line = colvar_array[x + 1]
    time = int((x + 1) * dt * hill_frequency)

    write_line = f"{time:15} {line[0]:20.16f} {line[1]:20.16f}          {sigma_proj}           {sigma_ext} {line[2]:20.16f}            {bias}\n"
    file.write(write_line)

    if time % 10000 == 0:
        print(f"Simulation time: {time} ps")
