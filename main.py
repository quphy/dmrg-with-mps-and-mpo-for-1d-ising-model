from mps import MPS
from mpo import MPO
from dmrg import DMRG
from idmrg import iDMRG
#param for ising model
L=8
d = 2
dim_max = 16
h = 0.5
J = 1.0
tol = 1e-10
max_sweeps = 20
max_steps = 30


print ("finite dmrg (obc)")
mps_obc = MPS(L, d, dim_max)
mpo_obc = MPO(L, d, h, J, periodic=False)
#begin sweeping
dmrg_obc = DMRG(mps_obc, mpo_obc, dim_max)
last_energy = float('inf')
for sweep in range(1, max_sweeps+1):
    dmrg_obc.left_to_right()
    dmrg_obc.right_to_left()
    current_energy = dmrg_obc.e[-1] 
    energy_diff = abs(current_energy - last_energy)
    if energy_diff < tol:
        print(f"dmrg converged at step {sweep}, energy: {current_energy}")
        break
    last_energy = current_energy

print(f"obc total energy: {dmrg_obc.e[-1]}")

print("finite dmrg (pbc)")
mps_pbc = MPS(L, d, dim_max)
mpo_pbc = MPO(L, d, h, J, periodic=True)
dmrg_pbc = DMRG(mps_pbc, mpo_pbc, dim_max)
for sweep in range(1, max_sweeps+1):
    dmrg_pbc.left_to_right()
    dmrg_pbc.right_to_left()
    current_energy = dmrg_pbc.e[-1] 
    energy_diff = abs(current_energy - last_energy)
    if energy_diff < tol:
        print(f"dmrg converged at step {sweep}, energy: {current_energy}")
        break
    last_energy = current_energy
print(f"pbc total energy: {dmrg_pbc.e[-1]}, energy per site: {dmrg_pbc.e[-1]/float(L)}")

print("infinite dmrg")
idmrg= iDMRG(d, h, J, dim_max, tol)
E_per_site = float('inf')
E_prev =0.0
for step in range(1, max_steps + 1):
    E_pre_persite = E_per_site
    E_current = idmrg.step()
    if step == 1:
        E_per_site = E_current / 2
    else:
        E_per_site = (E_current - E_prev) / 2
        E_prev = E_current 
    if step > 1 and abs(E_per_site - E_pre_persite) < tol:
        print(f"idmrg converged at step {step}, energy per site: {E_per_site}")
        energy_persite = E_per_site 
        break
print(f"energy per site: {energy_persite}")

