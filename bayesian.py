import numpy as np
import pandas as pd
import emcee
import astropy.cosmology as cosmo
import astropy.units as u
from scipy import integrate
from emcee.moves import StretchMove, WalkMove
import pickle
import multiprocessing
import time
from astropy.cosmology import FlatLambdaCDM
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Literal
from pathlib import Path
import warnings
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
from cosmographic_expansions.expansions import ExpansionModels
from cosmographic_expansions.transforms import CosmographicTransforms
from cosmographic_expansions.pade import PadeApproximants 
from cosmographic_expansions.expansions import EISModel, EISModelJAX
from cosmographic_expansions.transforms import EISTransforms
import scienceplots
from astropy.constants import c

import matplotlib.pyplot as plt
plt.style.use(['science', 'notebook', 'grid'])
numpyro.set_platform("gpu")

# True cosmology parameters
H0_true = 70.0
Om_true = 0.334
Ok_true = 0.0
w_true = -1.0

# Load Pantheon+ data
url = 'https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat'
df = pd.read_csv(url, delim_whitespace=True)
z = df['zCMB'].to_numpy()
mu_obs = df['MU_SH0ES'].to_numpy()
#dl = 10**(mu_obs/5 + 1) * u.pc.to(u.Mpc)
mu_err = df['MU_SH0ES_ERR_DIAG'].to_numpy()
cosmo = FlatLambdaCDM(H0=H0_true, Om0=Om_true)
dl = cosmo.luminosity_distance(z).value  # in Mpc

dl_err = (np.log(10)/5) * dl * mu_err



# Prior ranges
H0_range: Tuple[float, float] = (60.0, 80.0)
q0_range: Tuple[float, float] = (-5.0, 5.0)
j0_range: Tuple[float, float] = (-10.0, 10.0)
s0_range: Tuple[float, float] = (-10.0, 10.0)
c0_range: Tuple[float, float] = (-10.0, 20.0)
p0_range: Tuple[float, float] = (-30.0, 10.0)

# EIS parameter ranges
E1_range: Tuple[float, float] = (-1, 5)
E2_range: Tuple[float, float] = (-1, 5)
E3_range: Tuple[float, float] = (-5, 5)
E4_range: Tuple[float, float] = (-6, 5)
E5_range: Tuple[float, float] = (-5, 10.0)

# Convert to JAX arrays

z_jax = jnp.array(z)
dl_jax = jnp.array(dl)
dl_err_jax = jnp.array(dl_err)
c_kmps: float = c.to(u.km / u.s).value


# Initialize models
expansion_models = ExpansionModels()
pade_models = PadeApproximants()
eis_models = EISModelJAX()
eis_transforms = EISTransforms()
cosmo_transforms = CosmographicTransforms()



def get_prior_ranges(param_type='cosmographic'):
    """Get prior ranges for different parameter types"""
    if param_type == 'cosmographic':
        return [H0_range, q0_range, j0_range, s0_range, c0_range, p0_range]
    elif param_type == 'eis':
        return [H0_range, E1_range, E2_range, E3_range, E4_range, E5_range]
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def get_pade_functions(order: int) -> List:
    """Get appropriate Pade approximants for given order"""
    if order < 3:
        return []
    elif order == 3:
        return [('pade_21', pade_models.pade_21)]
    elif order == 4:
        return [('pade_22', pade_models.pade_22)]
    elif order == 5:
        return [('pade_32', pade_models.pade_32), ('pade_average', pade_models.pade_average)]
    else:  # order >= 6
        return [('pade_42', pade_models.pade_42), ('pade_51', pade_models.pade_51)]


# ==================== Numpyro Models ====================

def numpyro_z_series(z_jax, dl_jax, dl_err_jax, order, fixed_H0=None):
    """Z-series model"""
    prior_ranges = get_prior_ranges('cosmographic')
    param_names = ['H0', 'q0', 'j0', 's0', 'c0', 'p0']
    params = []
    for i in range(order):
        if i == 0 and fixed_H0 is not None:
            # Fix H0 to a specific value
            param = numpyro.deterministic(param_names[i], jnp.array(fixed_H0))
        else:
            param = numpyro.sample(param_names[i], dist.Uniform(prior_ranges[i][0], prior_ranges[i][1]))
        params.append(param)
    
    model_dl = jnp.array(expansion_models.z_series(z_jax, *params, order=order))
    numpyro.sample('obs', dist.Normal(model_dl, dl_err_jax), obs=dl_jax)


def numpyro_y_series(z_jax, dl_jax, dl_err_jax, order, fixed_H0=None):
    """Y-series model"""
    prior_ranges = get_prior_ranges('cosmographic')
    param_names = ['H0', 'q0', 'j0', 's0', 'c0', 'p0']
    params = []
    for i in range(order):
        if i == 0 and fixed_H0 is not None:
            param = numpyro.deterministic(param_names[i], jnp.array(fixed_H0))
        else:
            param = numpyro.sample(param_names[i], dist.Uniform(prior_ranges[i][0], prior_ranges[i][1]))
        params.append(param)
    
    y_jax = z_jax / (1 + z_jax)
    model_dl = jnp.array(expansion_models.y_series(y_jax, *params, order=order))
    numpyro.sample('obs', dist.Normal(model_dl, dl_err_jax), obs=dl_jax)


def numpyro_log_series(z_jax, dl_jax, dl_err_jax, order, fixed_H0=None):
    """Log-series model"""
    prior_ranges = get_prior_ranges('cosmographic')
    param_names = ['H0', 'q0', 'j0', 's0', 'c0', 'p0']
    params = []
    for i in range(order):
        if i == 0 and fixed_H0 is not None:
            param = numpyro.deterministic(param_names[i], jnp.array(fixed_H0))
        else:
            param = numpyro.sample(param_names[i], dist.Uniform(prior_ranges[i][0], prior_ranges[i][1]))
        params.append(param)
    
    log1z_jax = jnp.log(1 + z_jax)
    model_dl = jnp.array(expansion_models.log_series(log1z_jax, *params, order=order))
    numpyro.sample('obs', dist.Normal(model_dl, dl_err_jax), obs=dl_jax)


def numpyro_eis_series(z_jax, dl_jax, dl_err_jax, order, fixed_H0=None):
    """EIS-series model"""
    prior_ranges = get_prior_ranges('eis')
    param_names = ['H0', 'E1', 'E2', 'E3', 'E4', 'E5']
    params = []
    for i in range(order):
        if i == 0 and fixed_H0 is not None:
            param = numpyro.deterministic(param_names[i], jnp.array(fixed_H0))
        else:
            param = numpyro.sample(param_names[i], dist.Uniform(prior_ranges[i][0], prior_ranges[i][1]))
        params.append(param)
    
    model_dl = jnp.array(eis_models.luminosity_distance(z_jax, *params, order=order))
    numpyro.sample('obs', dist.Normal(model_dl, dl_err_jax), obs=dl_jax)


def numpyro_pade_model(z_jax, dl_jax, dl_err_jax, order, pade_func, fixed_H0=None):
    """Pade approximant model"""
    prior_ranges = get_prior_ranges('cosmographic')
    param_names = ['H0', 'q0', 'j0', 's0', 'c0', 'p0']
    params = []
    for i in range(order):
        if i == 0 and fixed_H0 is not None:
            param = numpyro.deterministic(param_names[i], jnp.array(fixed_H0))
        else:
            param = numpyro.sample(param_names[i], dist.Uniform(prior_ranges[i][0], prior_ranges[i][1]))
        params.append(param)
    
    model_dl = jnp.array(pade_func(z_jax, *params, order=0))
    numpyro.sample('obs', dist.Normal(model_dl, dl_err_jax), obs=dl_jax)


# ==================== Run MCMC ====================

def run_mcmc_for_series(model_func, series_name, order, fixed_H0=None, num_warmup=1000, num_samples=2000, num_chains=2):
    """Run MCMC for a given series and order"""
    h0_str = f" (H0 fixed at {fixed_H0})" if fixed_H0 is not None else ""
    print(f"\nRunning MCMC for {series_name}, order {order}{h0_str}")
    print(f"{'='*60}")
    
    nuts_kernel = NUTS(model_func)
    mcmc = MCMC(
        nuts_kernel, 
        num_warmup=num_warmup, 
        num_samples=num_samples, 
        num_chains=num_chains, 
        progress_bar=True, 
        chain_method='vectorized'
    )
    
    key = jax.random.PRNGKey(hash(f"{series_name}_{order}_{fixed_H0}") % 2**32)
    mcmc.run(key, z_jax, dl_jax, dl_err_jax, order, fixed_H0)
    mcmc.print_summary()
    
    return {
        'mcmc': mcmc,
        'idata': az.from_numpyro(mcmc),
        'series': series_name,
        'order': order,
        'fixed_H0': fixed_H0
    }


def run_mcmc_for_pade(pade_name, pade_func, order, fixed_H0=None, num_warmup=300, num_samples=2000, num_chains=5):
    """Run MCMC for Pade approximant"""
    h0_str = f" (H0 fixed at {fixed_H0})" if fixed_H0 is not None else ""
    print(f"\nRunning MCMC for {pade_name}, order {order}{h0_str}")
    print(f"{'='*60}")
    
    # Create a closure to pass the pade function
    def pade_model_wrapper(z_jax, dl_jax, dl_err_jax, order, fixed_H0):
        return numpyro_pade_model(z_jax, dl_jax, dl_err_jax, order, pade_func, fixed_H0)
    
    nuts_kernel = NUTS(pade_model_wrapper)
    mcmc = MCMC(
        nuts_kernel, 
        num_warmup=num_warmup, 
        num_samples=num_samples, 
        num_chains=num_chains, 
        progress_bar=True, 
        chain_method='vectorized'
    )
    
    key = jax.random.PRNGKey(hash(f"{pade_name}_{order}_{fixed_H0}") % 2**32)
    mcmc.run(key, z_jax, dl_jax, dl_err_jax, order, fixed_H0)
    mcmc.print_summary()
    
    return {
        'mcmc': mcmc,
        'idata': az.from_numpyro(mcmc),
        'series': pade_name,
        'order': order,
        'fixed_H0': fixed_H0
    }


# ==================== Main Analysis ====================

# Dictionary to store all results
all_results = {}

# Orders to analyze
orders = [3, 4, 5, 6]

# Set this to fix H0, or None to let it vary
# Example: fixed_H0_value = 70.0  # Fix H0 at 70
fixed_H0_value = None  # Let H0 vary freely

for order in orders:
    print(f"\n{'#'*70}")
    print(f"# ORDER {order}")
    print(f"{'#'*70}\n")
    
    order_results = {}
    
    # 1. Z-series
    order_results['z_series'] = run_mcmc_for_series(
        numpyro_z_series, 'z_series', order, fixed_H0=fixed_H0_value
    )
    
    # 2. Y-series
    order_results['y_series'] = run_mcmc_for_series(
        numpyro_y_series, 'y_series', order, fixed_H0=fixed_H0_value
    )
    
    # 3. Log-series
    order_results['log_series'] = run_mcmc_for_series(
        numpyro_log_series, 'log_series', order, fixed_H0=fixed_H0_value
    )
    
    # 4. EIS-series
    order_results['eis_series'] = run_mcmc_for_series(
        numpyro_eis_series, 'eis_series', order, fixed_H0=fixed_H0_value
    )
    
    # 5. Pade approximants
    pade_funcs = get_pade_functions(order)
    for pade_name, pade_func in pade_funcs:
        order_results[pade_name] = run_mcmc_for_pade(
            pade_name, pade_func, order, fixed_H0=fixed_H0_value
        )
    
    all_results[f'order_{order}'] = order_results

print(f"\n{'='*70}")
print("Analysis Complete!")
print(f"{'='*70}\n")

# ==================== Save Results ====================

import pickle
with open('bayesian_all_series_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print("Results saved to 'bayesian_all_series_results.pkl'")


# ==================== Generate Summary ====================

def generate_summary_statistics(all_results):
    """Generate summary statistics for all series and orders"""
    summary_data = []
    
    for order_key, order_results in all_results.items():
        order = int(order_key.split('_')[1])
        
        for series_name, result in order_results.items():
            idata = result['idata']
            
            # Get parameter names
            param_names = list(idata.posterior.data_vars.keys())
            
            # Calculate statistics for each parameter
            for param in param_names:
                samples = idata.posterior[param].values.flatten()
                
                summary_data.append({
                    'Order': order,
                    'Series': series_name,
                    'Parameter': param,
                    'Mean': np.mean(samples),
                    'Std': np.std(samples),
                    'Median': np.median(samples),
                    'HDI_2.5%': np.percentile(samples, 2.5),
                    'HDI_97.5%': np.percentile(samples, 97.5),
                })
    
    return pd.DataFrame(summary_data)


summary_df = generate_summary_statistics(all_results)
summary_df.to_csv('bayesian_summary_statistics.csv', index=False)
print("\nSummary statistics saved to 'bayesian_summary_statistics.csv'")
print(summary_df.head(20))



"""
######################################################################
# ORDER 3
######################################################################


Running MCMC for z_series, order 3
============================================================
sample: 100%|██████████████████████████████████████████████████████████████████| 3000/3000 [00:11<00:00, 270.92it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     69.84      0.29     69.84     69.34     70.31   1237.51      1.00
        j0      0.19      0.26      0.17     -0.24      0.60   1230.65      1.00
        q0     -0.41      0.06     -0.41     -0.50     -0.32   1139.68      1.00

Number of divergences: 0

Running MCMC for y_series, order 3
============================================================
sample: 100%|██████████████████████████████████████████████████████████████████| 3000/3000 [00:08<00:00, 335.89it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     62.66      0.17     62.66     62.36     62.92   2479.99      1.00
        j0     -9.90      0.10     -9.93    -10.00     -9.77   3285.36      1.00
        q0      4.22      0.05      4.22      4.15      4.30   2182.97      1.00

Number of divergences: 0

Running MCMC for log_series, order 3
============================================================
sample: 100%|██████████████████████████████████████████████████████████████████| 3000/3000 [00:13<00:00, 230.33it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     69.99      0.35     70.00     69.45     70.58   1011.11      1.00
        j0      0.64      1.28      0.61     -1.36      2.78    889.37      1.01
        q0     -0.48      0.12     -0.48     -0.68     -0.28    876.81      1.01

Number of divergences: 0

Running MCMC for eis_series, order 3
============================================================
sample: 100%|██████████████████████████████████████████████████████████████████| 3000/3000 [00:11<00:00, 270.73it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        E1      0.51      0.07      0.51      0.40      0.62   1064.06      1.00
        E2      0.67      0.29      0.66      0.19      1.11   1138.70      1.00
        H0     69.99      0.30     69.99     69.49     70.47   1290.53      1.00

Number of divergences: 0

Running MCMC for pade_21, order 3
============================================================
sample: 100%|███████████████████████████████████████████████████████████████████| 2300/2300 [02:57<00:00, 12.99it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     69.13      3.38     70.09     62.72     72.69      2.51     13.76
        j0      2.04      3.34      1.78     -2.67      7.32      2.73      3.53
        q0     -0.79      1.39     -0.56     -3.27      1.01       nan     19.00

Number of divergences: 163

######################################################################
# ORDER 4
######################################################################


Running MCMC for z_series, order 4
============================================================
sample: 100%|██████████████████████████████████████████████████████████████████| 3000/3000 [00:16<00:00, 185.56it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     69.98      0.33     69.98     69.46     70.56    882.10      1.00
        j0      0.96      0.81      0.95     -0.30      2.39    759.42      1.01
        q0     -0.49      0.10     -0.49     -0.65     -0.32    743.34      1.01
        s0      0.67      2.29      0.03     -2.22      3.94    646.25      1.01

Number of divergences: 0

Running MCMC for y_series, order 4
============================================================
sample: 100%|██████████████████████████████████████████████████████████████████| 3000/3000 [00:18<00:00, 165.06it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     69.53      0.34     69.53     68.95     70.07   1266.49      1.00
        j0     -5.49      1.79     -5.43     -8.51     -2.57   1059.61      1.00
        q0     -0.06      0.15     -0.06     -0.29      0.19   1028.99      1.00
        s0      1.01      5.75      1.43     -7.52      9.92   1764.62      1.00

Number of divergences: 0

Running MCMC for log_series, order 4
============================================================
sample: 100%|██████████████████████████████████████████████████████████████████| 3000/3000 [00:16<00:00, 181.38it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     69.99      0.33     70.00     69.45     70.54   1078.23      1.00
        j0      0.94      0.90      0.99     -0.45      2.36    801.22      1.00
        q0     -0.49      0.09     -0.50     -0.64     -0.34    874.62      1.00
        s0     -0.55      5.80     -0.81    -10.00      7.60   1076.60      1.00

Number of divergences: 0

Running MCMC for eis_series, order 4
============================================================
sample: 100%|██████████████████████████████████████████████████████████████████| 3000/3000 [00:14<00:00, 200.48it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        E1      0.51      0.10      0.51      0.34      0.67    811.66      1.00
        E2      0.61      0.72      0.60     -0.58      1.77    796.36      1.00
        E3      0.29      1.69      0.21     -2.42      3.08    856.61      1.00
        H0     70.00      0.33     70.00     69.43     70.51   1081.71      1.00

Number of divergences: 34

Running MCMC for pade_22, order 4
============================================================
sample: 100%|███████████████████████████████████████████████████████████████████| 2300/2300 [03:04<00:00, 12.47it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     72.94      4.02     72.29     68.40     80.00      2.50     30.49
        j0     -4.07      3.97     -4.57    -10.00      1.33      2.52     11.38
        q0     -1.03      1.05     -0.52     -2.31      0.35       nan     28.17
        s0      1.05      5.46      1.55     -4.80      9.99      2.88      2.72

Number of divergences: 640

######################################################################
# ORDER 5
######################################################################


Running MCMC for z_series, order 5
============================================================
sample: 100%|██████████████████████████████████████████████████████████████████| 3000/3000 [00:23<00:00, 128.31it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     69.92      0.28     69.92     69.45     70.38   1146.65      1.00
        c0      9.09      5.14      8.21      3.10     19.62    654.68      1.00
        j0      0.57      0.63      0.62     -0.34      1.62    511.37      1.00
        q0     -0.45      0.07     -0.46     -0.56     -0.34    519.09      1.00
        s0     -0.99      2.72     -1.04     -5.36      3.34    698.71      1.00

Number of divergences: 0

Running MCMC for y_series, order 5
============================================================
sample: 100%|██████████████████████████████████████████████████████████████████| 3000/3000 [00:21<00:00, 140.46it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     69.74      0.34     69.74     69.17     70.27   1345.28      1.00
        c0      4.83      8.61      4.66     -7.92     19.08   2306.65      1.00
        j0     -0.95      1.12     -0.93     -2.88      0.83   1149.03      1.00
        q0     -0.31      0.11     -0.31     -0.51     -0.13   1126.78      1.00
        s0      1.42      5.68      2.06     -7.03      9.91   2087.47      1.00

Number of divergences: 0

Running MCMC for log_series, order 5
============================================================
sample: 100%|██████████████████████████████████████████████████████████████████| 3000/3000 [00:23<00:00, 128.73it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     69.97      0.32     69.96     69.45     70.48   1284.69      1.00
        c0      5.31      8.51      5.61     -6.94     19.74   2192.71      1.00
        j0      0.96      1.14      0.92     -0.89      2.74   1020.18      1.00
        q0     -0.49      0.11     -0.48     -0.65     -0.31   1003.00      1.00
        s0     -0.26      4.67      0.06     -8.24      6.97   1121.54      1.00

Number of divergences: 0

Running MCMC for eis_series, order 5
============================================================
sample: 100%|██████████████████████████████████████████████████████████████████| 3000/3000 [00:19<00:00, 151.69it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        E1      0.49      0.11      0.49      0.36      0.67     13.58      1.06
        E2      0.75      0.81      0.79     -0.63      1.78     18.75      1.05
        E3      0.09      2.20     -0.33     -2.99      4.14     80.18      1.02
        E4     -0.83      3.13     -1.24     -4.40      4.84     25.13      1.04
        H0     70.04      0.31     70.07     69.48     70.45     31.76      1.03

Number of divergences: 860

Running MCMC for pade_32, order 5
============================================================
sample: 100%|███████████████████████████████████████████████████████████████████| 2300/2300 [05:29<00:00,  6.98it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     67.69      5.33     70.10     60.01     74.78       nan     66.28
        c0      8.14      5.70      6.27      1.39     15.08      2.69      4.09
        j0     -0.01      2.93      2.01     -4.97      2.44       nan     41.29
        q0      0.23      1.91     -0.57     -2.13      3.51      2.50    179.68
        s0      1.39      7.41      6.79     -9.53      8.41       nan     26.66

Number of divergences: 2608

Running MCMC for pade_average, order 5
============================================================
sample: 100%|███████████████████████████████████████████████████████████████████| 2300/2300 [05:25<00:00,  7.08it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     66.30      6.45     64.02     60.00     76.70      2.51     24.24
        c0      3.08     10.34     -4.00     -7.22     16.99       nan      5.53
        j0     -0.65      1.55     -0.60     -2.21      1.88      2.91      2.83
        q0      0.55      2.34      0.98     -3.35      2.88      2.50     61.94
        s0     -3.64      6.15     -5.00    -10.00      5.96       nan     31.56

Number of divergences: 3651

######################################################################
# ORDER 6
######################################################################


Running MCMC for z_series, order 6
============================================================
sample: 100%|███████████████████████████████████████████████████████████████████| 3000/3000 [00:53<00:00, 56.26it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     69.94      0.29     69.94     69.46     70.38   1427.90      1.00
        c0      4.91      6.77      4.12     -4.29     18.80    992.14      1.01
        j0      0.63      0.52      0.62     -0.24      1.54    753.04      1.01
        p0    -10.65     11.42    -11.04    -29.88      5.53   2190.78      1.00
        q0     -0.46      0.07     -0.47     -0.57     -0.35    839.72      1.00
        s0     -1.53      3.00     -1.49     -6.56      2.89   1039.86      1.00

Number of divergences: 42

Running MCMC for y_series, order 6
============================================================
sample: 100%|██████████████████████████████████████████████████████████████████| 3000/3000 [00:26<00:00, 113.73it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     69.85      0.33     69.85     69.31     70.38   1491.68      1.00
        c0      4.54      8.56      4.36    -10.00     16.64   3444.07      1.00
        j0      0.36      1.14      0.43     -1.55      2.10   1147.88      1.00
        p0     -9.74     11.55     -9.60    -26.04      9.61   2905.54      1.00
        q0     -0.41      0.11     -0.41     -0.60     -0.24   1119.13      1.00
        s0      1.84      5.55      2.55     -6.39     10.00   1565.73      1.00

Number of divergences: 0

Running MCMC for log_series, order 6
============================================================
sample: 100%|██████████████████████████████████████████████████████████████████| 3000/3000 [00:23<00:00, 129.51it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     69.94      0.31     69.94     69.44     70.48   1934.65      1.00
        c0      6.45      8.53      7.28     -6.00     19.99   2141.34      1.00
        j0      0.98      0.95      1.08     -0.58      2.42   1166.34      1.00
        p0    -10.01     11.42    -10.09    -29.58      5.94   2958.09      1.00
        q0     -0.48      0.10     -0.48     -0.64     -0.32   1318.49      1.00
        s0      0.54      3.95      0.37     -6.25      6.94   1051.56      1.00

Number of divergences: 0

Running MCMC for eis_series, order 6
============================================================
sample: 100%|███████████████████████████████████████████████████████████████████| 3000/3000 [02:41<00:00, 18.58it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        E1      0.30      0.12      0.29      0.10      0.51      1.37      1.75
        E2      2.34      0.92      2.40      0.95      3.81      1.27      1.94
        E3     -2.52      1.76     -3.09     -4.94      0.39      1.82      1.43
        E4     -4.15      1.85     -4.89     -6.00     -1.42      2.14      1.34
        E5     -2.59      2.66     -3.63     -5.00      2.35      2.55      1.26
        H0     70.38      0.35     70.39     69.81     70.97      2.78      1.21

Number of divergences: 1092

Running MCMC for pade_42, order 6
============================================================
sample: 100%|███████████████████████████████████████████████████████████████████| 2300/2300 [07:48<00:00,  4.91it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     69.91      0.58     70.08     68.90     70.41      3.00      2.39
        c0      7.29      7.70     12.27     -9.75     13.12      6.91      1.24
        j0      0.73      1.92      1.23     -2.73      2.74       nan      2.51
        p0    -13.94      9.85    -18.07    -29.33      3.27     13.24      1.11
        q0     -0.45      0.25     -0.56     -0.66      0.02       nan      3.24
        s0      0.22      3.86      1.12     -5.47      6.83     10.76      1.15

Number of divergences: 1168

Running MCMC for pade_51, order 6
============================================================
sample: 100%|███████████████████████████████████████████████████████████████████| 2300/2300 [05:00<00:00,  7.66it/s]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        H0     73.52      3.10     75.19     69.62     77.26       nan     16.49
        c0      3.91      6.07      5.03     -7.33     12.41      7.65      1.25
        j0      1.37      3.63      0.50     -4.00      6.59       nan      8.95
        p0     -8.48      9.60    -10.18    -27.00      3.38       nan      1.32
        q0     -1.21      1.08     -0.59     -3.23     -0.36      2.51     21.01
        s0      0.74      5.35      0.52     -5.65      7.21      2.80      3.08

Number of divergences: 3532

======================================================================
Analysis Complete!
======================================================================

"""