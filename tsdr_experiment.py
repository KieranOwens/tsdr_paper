##########################
# import packages
##########################

# scientific computing
import numpy as np
import scipy
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, Isomap
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern']
import seaborn as sb
ICEFIRE = sb.color_palette('icefire', as_cmap=True)

# tsdr methods
from tsdr.slowness import SFA, BioSFA
from tsdr.autocorrelation import TICA, TMCA, CSA, TCCA_Stiefel, sPCA_dwt
from tsdr.predictability import TLPC, PrCA, DiCCA
from tsdr.determinism import DyCA, DMD
from tsdr.nonstationarity import ASSA, SSAsir, SSAsave, SSAcor, WaSSAf, WaSSAr, BSSnonstat
from tsdr.diffusion import DMcov, DIG
from tsdr.latentvariable import LDS

##########################
# Experiment settings
##########################

# Colour palette
PALETTE = {'black': '#666666',
           'red': '#C23B23',
           'orange': '#c8760b', #F39A27',
           'yellow': '#d3c01a',#'#EADA52',
           'green': '#02902d',#'#03C03C',
           'blue': '#3a7696',#'#579ABE',
           'purple': '#6431b2', #'#976ED7',
           'pink': '#e441a8'}#'#E68C96'}
DPI = 500

# Convenience function to construct a simple autoencoder
def encoder(encoder_weights, encoder_biases, data):
    res_ae = data
    for index, (w, b) in enumerate(zip(encoder_weights, encoder_biases)):
        if index+1 == len(encoder_weights):
            res_ae = res_ae@w+b 
        else:
            res_ae = np.maximum(0, res_ae@w+b)
    return res_ae

# Method names and dictionary
method_names = ['PCA', 'FastICA', 'tSNE', 'Isomap', 'AE',
                'SFA', 'BioSFA',
                'TICA', 'TMCA', 'TCCA', 'CSA', 'sPCA', 
                'TLPC', 'PrCA', 'DiCCA',
                'DMD', 'DyCA',
                'ASSA', 'SSAsir', 'SSAsave', 'SSAcor', 'WaSSAf', 'WaSSAr', 'BSSnonstat',
                'DMcov', 'DIG',
                'DFA']

methods = {'PCA': {'f': PCA,
                   'color': PALETTE['black'],
                   'kwargs': {'n_components':1}},

            'FastICA': {'f': FastICA,
                   'color': PALETTE['black'],
                   'kwargs': {'n_components':1}},

            'tSNE': {'f': TSNE,
                   'color': PALETTE['black'],
                   'kwargs': {'n_components':1}},

            'Isomap': {'f': Isomap,
                   'color': PALETTE['black'],
                   'kwargs': {'n_components':1}},

            'AE': {'f': MLPRegressor,
                'color': PALETTE['black'],
                'kwargs': {'alpha':1e-15, 
                                'hidden_layer_sizes':(10, 10, 1, 10, 10), 
                                'random_state':0, 
                                'max_iter':10000}},
            
            'SFA': {'f': SFA,
                    'color': PALETTE['red'],
                    'kwargs': {'n_components':1}},

            'BioSFA': {'f': BioSFA,
                    'color': PALETTE['red'],
                    'kwargs': {'n_components':1, 'n_iters':1000, 'lr':0.1}},

            'TICA': {'f': TICA,
                    'color': PALETTE['orange'],
                    'kwargs': {'n_components':1}},

            'TMCA': {'f': TMCA,
                    'color': PALETTE['orange'],
                    'kwargs': {'n_components':1}},

            'TCCA': {'f': TCCA_Stiefel,
                    'color': PALETTE['orange'],
                    'kwargs': {'n_components':1}},

            'CSA': {'f': CSA,
                    'color': PALETTE['orange'],
                    'kwargs': {'n_components':1, 'lags':10}},

            'sPCA': {'f': sPCA_dwt,
                    'color': PALETTE['orange'],
                    'kwargs': {'n_components':1}},

            'TLPC': {'f': TLPC,
                    'color': PALETTE['yellow'],
                    'kwargs': {'n_components':1}},

            'PrCA': {'f': PrCA,
                    'color': PALETTE['yellow'],
                    'kwargs': {'n_components':1,'p':5, 'method':'linear_reg'}},

            'DiCCA': {'f': DiCCA,
                    'color': PALETTE['yellow'],
                    'kwargs': {'n_components':1, 's':5}},
        
            'DMD': {'f': DMD,
                    'color': PALETTE['green'],
                    'kwargs': {'n_components':1}},

            'DyCA': {'f': DyCA,
                    'color': PALETTE['green'],
                    'kwargs': {'n_components':1}},

            'ASSA': {'f': ASSA,
                   'color': PALETTE['blue'],
                   'kwargs': {'n_components':1}},

            'SSAsir': {'f': SSAsir,
                   'color': PALETTE['blue'],
                   'kwargs': {'n_components':1}},

            'SSAsave': {'f': SSAsave,
                   'color': PALETTE['blue'],
                   'kwargs': {'n_components':1}},

            'SSAcor': {'f': SSAcor,
                   'color': PALETTE['blue'],
                   'kwargs': {'n_components':1}},

            'WaSSAf': {'f': WaSSAf,
                   'color': PALETTE['blue'],
                   'kwargs': {'n_components':1}},

            'WaSSAr': {'f': WaSSAr,
                   'color': PALETTE['blue'],
                   'kwargs': {'n_components':1}},

            'BSSnonstat': {'f': BSSnonstat,
                   'color': PALETTE['blue'],
                   'kwargs': {'n_components':1, 'lag':1, 'max_iters':10}},

            'DMcov': {'f': DMcov,
                    'color': PALETTE['purple'],
                    'kwargs': {'n_components':1, 'window':100, 'step':10, 
                               'distance':'mahalanobis', 'normalise':False}},

            'DIG': {'f': DIG,
                    'color': PALETTE['purple'],
                    'kwargs': {'n_components':1, 'window':100, 'step':10}},

            'DFA': {'f': LDS,
                    'color': PALETTE['pink'],
                    'kwargs': {'n_components':1, 'constrain_A_identity':True}},             
        }

# Experiment variables
NOISE = [10**i for i in np.arange(-1, 7.5)]#[10**i for i in np.arange(-1, 15.5)]

ITERS = 100

R2 = np.zeros((len(method_names), len(NOISE), ITERS))

SEED = 0

RNG = np.random.default_rng(SEED)

D = 10

np.random.seed(seed=SEED)
A_LIST = [scipy.stats.ortho_group.rvs(D) for _ in range(ITERS)]


##########################
# time series for visualisation
##########################

Tmax = 1000
T = np.linspace(0, 4* np.pi, Tmax)
Z = np.sin(T)

var = 0.5
Y = np.concatenate([Z.reshape(-1,1), RNG.normal(scale=var, size=((len(T), D-1)))], axis=1)
X = Y @ scipy.stats.ortho_group.rvs(Y.shape[1], random_state=1)

var2 = 1.0
Y2 = np.concatenate([Z.reshape(-1,1), RNG.normal(scale=var2, size=((len(T), D-1)))], axis=1)
X2 = Y2 @ scipy.stats.ortho_group.rvs(Y.shape[1], random_state=1)

##########################
# experiment loop
##########################

z = Z.flatten()
z_window = np.array([np.mean(Z[step:step+100]) for step in range(0, len(Z) - 100, 10)]).flatten()

for k in range(ITERS):

    print(f'Iteration {k}...')

    A = A_LIST[k]

    for j, variance in enumerate(NOISE):

        print(f'... variance {variance}.')
        
        Ytemp = np.concatenate([Z.reshape(-1,1), RNG.normal(scale=variance, size=((len(T), D-1)))], axis=1)
        Xtemp = Ytemp @ A

        for i, method in enumerate(method_names):
            
            method_dict = methods[method]
            f, c, kwargs = method_dict['f'], method_dict['color'], method_dict['kwargs']

            if method in ['PCA', 'FastICA', 'tSNE', 'Isomap', 'AE']:
                Xtemp_ = Xtemp
            else:
                Xtemp_ = PCA(whiten=True).fit_transform(Xtemp)

            if method in ['AE']:
                AE = f(**kwargs).fit(Xtemp_, Xtemp_)
                weights = AE.coefs_
                biases = AE.intercepts_
                encoder_weights = weights[0:3]
                encoder_biases = biases[0:3]
                f_z = encoder(encoder_weights, encoder_biases, Xtemp_)
                R2[i,j,k] = np.corrcoef(f_z.flatten(), z)[0,1]**2
            else:
                try:
                    f_z = f(**kwargs).fit_transform(Xtemp_)

                    if method in ['DMcov', 'DIG']:

                        R2[i,j,k] = np.corrcoef(f_z.flatten(), z_window)[0,1]**2

                    else:

                        R2[i,j,k] = np.corrcoef(f_z.flatten(), z)[0,1]**2
                except:
                    print(f'Failed for method {method}')

# Save the results
np.savez('tsdr_experiment_results.npz', R2=R2, method_names=method_names, noise=NOISE, seed=SEED, rng_state=RNG.bit_generator.state)


##########################
# plot results
##########################

# set up figure
fig = plt.figure(figsize=(10, 7), layout="constrained")
mosaic = """
    AC.
    XZ.
    BDV
    """
ax = fig.subplot_mosaic(mosaic, height_ratios=[0.15, 0.03, 0.7])

N = len(method_names)
fontsize=14
lw=1.5

ax["B"].set_ylim((0,N))
ax["B"].set_yticks([n + 0.5 for n in range(N)][::-1])
ax["B"].set_yticklabels(method_names, fontsize=fontsize)

ax["D"].set_ylim((0,N))
ax["D"].set_yticks([n + 0.5 for n in range(N)][::-1])
ax["D"].set_yticklabels(method_names, fontsize=fontsize, weight='bold')

im0 = ax['V'].imshow(np.mean(R2, axis=2)[:,:9], vmin=0, vmax=1, aspect=1, cmap='inferno')
cbar = plt.colorbar(im0, ax=ax["V"], location='right', shrink=0.5)
cbar.set_label(label=r'$R^2$', rotation=0)
ax["V"].set_yticks(range(len(method_names)))
ax["V"].set_yticklabels(method_names, fontsize=fontsize, weight='bold')
ax["V"].set_xlabel(r'$\sigma^2$')
sigmas = [r'$10^{s}$'.replace('s', f'{int(i)}') for i in np.log10(NOISE)]
ax["V"].set_xticks([0, 2, 4, 6, 8])
ax["V"].set_xticklabels([sigmas[0], sigmas[2], sigmas[4], sigmas[6], sigmas[8]])

ax['A'].sharex(ax['B'])
im1 = ax["A"].imshow(X.T, aspect=30, vmin=-0.5, vmax=0.5, cmap=ICEFIRE)
plt.colorbar(im1, ax=ax["A"], location='right', shrink=0.6, aspect=10)
ax["A"].set_title(r'$\sigma^2 = 0.5$')

ax['C'].sharex(ax['D'])
im2 = ax["C"].imshow(X2.T, aspect=30, vmin=-1, vmax=1, cmap=ICEFIRE)
plt.colorbar(im2, ax=ax["C"], location='right', shrink=0.6, aspect=10)
ax["C"].set_title(r'$\sigma^2 = 1.0$')

ax['X'].sharex(ax['B'])
ax['Z'].sharex(ax['D'])
for loc in "XZ":
    ax[loc].spines['top'].set_visible(False)
    ax[loc].spines['right'].set_visible(False)
    ax[loc].plot(Z, c=PALETTE['black'], lw=1.5)
    ax[loc].set_xlabel('Time')
    ax[loc].set_ylim((-1.2, 1.2))
    ax[loc].set_yticks([-1,1])
    ax[loc].set_ylabel('z')

for loc in "AC":
    ax[loc].set_yticks([])
    ax[loc].set_xlabel('Time')
    ax[loc].set_ylabel('Variables')

for loc in "BD":
    ax[loc].spines['top'].set_visible(False)
    ax[loc].spines['right'].set_visible(False)
    ax[loc].spines['left'].set_visible(False)
    ax[loc].set_xlabel('Time')
    ax[loc].yaxis.set_tick_params(length=0)

fontsize=18
fig.text(0.02, 0.98, r'$\textbf{a}$', fontsize=fontsize)
fig.text(0.02, 0.84, r'$\textbf{b}$', fontsize=fontsize)
fig.text(0.02, 0.73, r'$\textbf{c}$', fontsize=fontsize)

fig.text(0.37, 0.98, r'$\textbf{d}$', fontsize=fontsize)
fig.text(0.37, 0.84, r'$\textbf{e}$', fontsize=fontsize)
fig.text(0.37, 0.73, r'$\textbf{f}$', fontsize=fontsize)

fig.text(0.7, 0.73, r'$\textbf{g}$', fontsize=fontsize)

for i, method in enumerate(method_names):
    method_dict = methods[method]
    f, c, kwargs = method_dict['f'], method_dict['color'], method_dict['kwargs']

    if method in ['AE']:
        AE = f(**kwargs).fit(X, X)
        weights = AE.coefs_
        biases = AE.intercepts_
        encoder_weights = weights[0:3]
        encoder_biases = biases[0:3]
        zae = encoder(encoder_weights, encoder_biases, X)
    else:
        zae = f(**kwargs).fit_transform(X)
    if method in ['DMcov', 'DIG']:
        zae = np.interp(x=np.arange(0, len(zae), len(zae)/Tmax), 
                      xp=np.arange(len(zae)), 
                      fp=list(zae.flatten())).reshape(-1,1)
    zae = np.sign(np.dot(zae.flatten(),Z.flatten())) * zae
    zae = MinMaxScaler((0.2, 0.8)).fit_transform(zae)
    ax["B"].plot(N - 1 - i + zae, color=c, lw=lw)
    ax["B"].get_yticklabels()[i].set_color(c)

    if method in ['AE']:
        AE = f(**kwargs).fit(X2, X2)
        weights = AE.coefs_
        biases = AE.intercepts_
        encoder_weights = weights[0:3]
        encoder_biases = biases[0:3]
        zae = encoder(encoder_weights, encoder_biases, X2)
    else:
        zae = f(**kwargs).fit_transform(X2)
    if method in ['DMcov', 'DIG']:
        zae = np.interp(x=np.arange(0, len(zae), 
                                  len(zae)/Tmax), xp=np.arange(len(zae)), 
                                  fp=list(zae.flatten())).reshape(-1,1)
    zae = np.sign(np.dot(zae.flatten(),Z.flatten())) * zae
    zae = MinMaxScaler((0.2, 0.8)).fit_transform(zae)
    ax["D"].plot(N - 1 - i + zae, color=c, lw=lw)
    ax["D"].get_yticklabels()[i].set_color(c)

    ax["V"].get_yticklabels()[i].set_color(c)

# save figure
fig.savefig('./tsdr_comparison.pdf', format='pdf', dpi=DPI, bbox_inches='tight')