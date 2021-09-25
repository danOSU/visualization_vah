# %%
import streamlit as st
import numpy as np
import time
import os
import subprocess
import matplotlib
import altair as alt
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.decomposition import PCA
from numpy.linalg import inv
import sklearn, matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process import kernels as krnl
import seaborn as sns

npc=3

# %%
# https://gist.github.com/beniwohli/765262
greek_alphabet_inv = { u'\u0391': 'Alpha', u'\u0392': 'Beta', u'\u0393': 'Gamma', u'\u0394': 'Delta', u'\u0395': 'Epsilon', u'\u0396': 'Zeta', u'\u0397': 'Eta', u'\u0398': 'Theta', u'\u0399': 'Iota', u'\u039A': 'Kappa', u'\u039B': 'Lamda', u'\u039C': 'Mu', u'\u039D': 'Nu', u'\u039E': 'Xi', u'\u039F': 'Omicron', u'\u03A0': 'Pi', u'\u03A1': 'Rho', u'\u03A3': 'Sigma', u'\u03A4': 'Tau', u'\u03A5': 'Upsilon', u'\u03A6': 'Phi', u'\u03A7': 'Chi', u'\u03A8': 'Psi', u'\u03A9': 'Omega', u'\u03B1': 'alpha', u'\u03B2': 'beta', u'\u03B3': 'gamma', u'\u03B4': 'delta', u'\u03B5': 'epsilon', u'\u03B6': 'zeta', u'\u03B7': 'eta', u'\u03B8': 'theta', u'\u03B9': 'iota', u'\u03BA': 'kappa', u'\u03BB': 'lamda', u'\u03BC': 'mu', u'\u03BD': 'nu', u'\u03BE': 'xi', u'\u03BF': 'omicron', u'\u03C0': 'pi', u'\u03C1': 'rho', u'\u03C3': 'sigma', u'\u03C4': 'tau', u'\u03C5': 'upsilon', u'\u03C6': 'phi', u'\u03C7': 'chi', u'\u03C8': 'psi', u'\u03C9': 'omega', }
greek_alphabet = {v: k for k, v in greek_alphabet_inv.items()}

# %%
zeta_over_s_str=greek_alphabet['zeta']+'/s(T)'
eta_over_s_str=greek_alphabet['eta']+'/s(T)'

# %%
v2_str='v'u'\u2082''{2}'
v3_str='v'u'\u2083''{2}'
v4_str='v'u'\u2084''{2}'


# %%
short_names = {
                'norm' : r'Energy Normalization', #0
                'trento_p' : r'TRENTo Reduced Thickness', #1
                'sigma_k' : r'Multiplicity Fluctuation', #2
                'nucleon_width' : r'Nucleon width [fm]', #3
                'dmin3' : r'Min. Distance btw. nucleons cubed [fm^3]', #4
                'Tswitch' : 'Particlization temperature [GeV]', #16
                'eta_over_s_T_kink_in_GeV' : r'Temperature of shear kink [GeV]', #7
                'eta_over_s_low_T_slope_in_GeV' : r'Low-temp. shear slope [GeV^-1]', #8
                'eta_over_s_high_T_slope_in_GeV' : r'High-temp shear slope [GeV^-1]', #9
                'eta_over_s_at_kink' : r'Shear viscosity at kink', #10
                'zeta_over_s_max' : r'Bulk viscosity max.', #11
                'zeta_over_s_T_peak_in_GeV' : r'Temperature of max. bulk viscosity [GeV]', #12
                'zeta_over_s_width_in_GeV' : r'Width of bulk viscosity [GeV]', #13
                'zeta_over_s_lambda_asymm' : r'Skewness of bulk viscosity', #14
                'initial_pressure_ratio' : r'R',
}


# %%
system_observables = {
                    'Pb-Pb-2760' : ['dNch_deta', 'dET_deta', 'dN_dy_pion', 'dN_dy_kaon' ,'dN_dy_proton', 'mean_pT_pion','mean_pT_kaon', 'mean_pT_proton']
                    }

# %%
obs_lims = {'dNch_deta': 2000. , 'dET_deta' : 2000. , 'dN_dy_pion' : 2000., 'dN_dy_kaon' : 500., 'dN_dy_proton' : 100., 'mean_pT_pion' : 1., 'mean_pT_kaon' : 1., 'mean_pT_proton' : 2.}

# %%
obs_word_labels = {
                    'dNch_deta' : r'Charged multiplicity',
                    'dN_dy_pion' : r'Pion dN/dy',
                    'dN_dy_kaon' : r'Kaon dN/dy',
                    'dN_dy_proton' : r'Proton dN/dy',
                    'dN_dy_Lambda' : r'Lambda dN/dy',
                    'dN_dy_Omega' : r'Omega dN/dy',
                    'dN_dy_Xi' : r'Xi dN/dy',
                    'dET_deta' : r'Transverse energy [GeV]',
                    'mean_pT_pion' : r'Pion mean pT [GeV]',
                    'mean_pT_kaon' : r'Kaon mean pT [GeV]',
                    'mean_pT_proton' : r'Proton mean pT [GeV]',
}

# %%
system = 'Pb-Pb-2760'

# %%
@st.cache(allow_output_mutation=True, show_spinner=False)
def load_emu():
    #load the emulator
    with open('PbPb2760_vah_emulators.dat',"rb") as f:
        emu=pickle.load(f)
    return emu

# %%
def predict_observables(model_parameters, Emulators, inverse_tf_matrix, SS_mean):
    mean=[]
    variance=[]
    theta=np.array(model_parameters).flatten()

    if len(theta)!=15:
        raise TypeError('The input model_parameters array does not have the right dimensions')
    else:
        theta=np.array(theta).reshape(1,15)
        for i in range(0,npc):
            mn,std=Emulators[i].predict(theta,return_std=True)
            mean.append(mn)
            variance.append(std**2)
    mean=np.array(mean).reshape(1,-1)
    inverse_transformed_mean=mean @ inverse_tf_matrix + np.array(SS_mean).reshape(1,-1)
    variance_matrix=np.diag(np.array(variance).flatten())
    A_p=inverse_tf_matrix
    inverse_transformed_variance=np.einsum('ik,kl,lj-> ij', A_p.T, variance_matrix, A_p, optimize=False)
    return inverse_transformed_mean, inverse_transformed_variance


# %%
@st.cache(allow_output_mutation=True, show_spinner=False)
def emu_predict(emu, params,inverse_tf_matrix, SS_mean):
    start = time.time()
    Yemu_cov = 0
    #Yemu_mean = emu.predict( np.array( [params] ), return_cov=False )
    Yemu_mean, Yemu_cov = predict_observables( np.array( [params] ), emu ,inverse_tf_matrix, SS_mean)
    end = time.time()
    time_emu = end - start
    return Yemu_mean, Yemu_cov, time_emu

# %%
ALICE_cent_bins = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]])

obs_cent_list = {
'Pb-Pb-2760': {
    'dNch_deta' : ALICE_cent_bins,
    'dET_deta' : np.array([[0, 2.5], [2.5, 5], [5, 7.5], [7.5, 10],
                           [10, 12.5], [12.5, 15], [15, 17.5], [17.5, 20],
                           [20, 22.5], [22.5, 25], [25, 27.5], [27.5, 30],
                           [30, 32.5], [32.5, 35], [35, 37.5], [37.5, 40],
                           [40, 45], [45, 50], [50, 55], [55, 60],
                           [60, 65], [65, 70]]), # 22 bins
    'dN_dy_pion'   : ALICE_cent_bins,
    'dN_dy_kaon'   : ALICE_cent_bins,
    'dN_dy_proton' : ALICE_cent_bins,
    'dN_dy_Lambda' : np.array([[0,5],[5,10],[10,20],[20,40],[40,60]]), # 5 bins
    'dN_dy_Omega'  : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'dN_dy_Xi'     : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'mean_pT_pion'   : ALICE_cent_bins,
    'mean_pT_kaon'   : ALICE_cent_bins,
    'mean_pT_proton' : ALICE_cent_bins,
    'pT_fluct' : np.array([[0,5],[5,10],[10,15],[15,20], [20,25],[25,30],[30,35],[35,40], [40,45],[45,50],[50,55],[55,60]]), #12 bins
    'v22' : ALICE_cent_bins,
    'v32' : np.array([[0,5],[5,10],[10,20],[20,30], [30,40],[40,50]]), # 6 bins
    'v42' : np.array([[0,5],[5,10],[10,20],[20,30], [30,40],[40,50]]), # 6 bins
    }
}


# %%
def main():
    st.title('Relativistic Heavy Ion Collisions Simulation with Viscous\
    Anisotropic Hydrodynamics')
    st.markdown('### Initial Data exploration and Visualization')
    st.markdown('This widget is inspired from a similar [widget]\
    (https://jetscape.org/sims-widget/) made by [Derek Everett](https://www.linkedin.com/in/derekseverett/).')
    st.markdown('The experimentally measured observables by the \
    [ALICE collaboration](https://home.cern/science/experiments/alice) \
    are shown as red dots.')
  #  st.markdown('The last row displays the temperature dependence of \
  #  the specific shear and bulk viscosities (red lines), \
  #  as determined by different parameters on the left sidebar.')
   # st.markdown('By default, these parameters are assigned the values\
   #  that fit the experimental data *best* (Best Guess).')

#    st.markdown(r'An important modelling ingredient is the\
#     particlization model used to convert hydrodynamic fields \
#     into individual hadrons. Three different viscous correction models \
#     can be selected by clicking the "Particlization model" button below.')

#    idf_names = ['Grad', 'Chapman-Enskog R.T.A', 'Pratt-Torrieri-Bernhard']
#    idf_name = st.selectbox('Particlization model',idf_names)

    # Reset button
#    st.markdown('<a href="javascript:window.location.href=window.location.href">Reset</a>', unsafe_allow_html=True)


#    inverted_idf_label = dict([[v,k] for k,v in idf_label.items()])
#    idf = inverted_idf_label[idf_name]

    #load the design
    #design, labels, design_max, design_min = load_design(system)
    inverse_tf_matrix = np.load('INV_MAT.npy')
    SS_mean = np.load('SS_MN.npy')
    load_prior = pd.read_csv('priorVAH.csv', index_col=0)
    maxar = load_prior.loc['max'].values
    minar = load_prior.loc['min'].values
    #design_min = [10, -0.7, 0.5, 0, 0.3, 0.135, 0.13, 0.01, -2, -1, 0.01, 0.12, 0.025, -0.8, 0.3]
    #design_max = [30, 0.7, 1.5, 4.91, 2, 0.165, 0.3, 0.2, 1, 2, 0.25, 0.3, 0.15, 0.8, 1]
    #load the emu
    emu = load_emu()

    #load the exp obs
    experiment=pd.read_csv(filepath_or_buffer="PbPb2760_experiment",index_col=0)
    y_exp= experiment.values[0,0:-32]
    y_exp_variance= experiment.values[1,0:-32]
    observables = experiment.keys()
    exp_label=[]
    nobs = len(observables)
    #observables, nobs, Yexp
    #initialize parameters
    #params_0 = (design_min+design_max)/2
    #print(params_0)
    params = []
    #print(design_min)
    #print(design_max)
    #updated params
    MAP = np.array([1.59767621e+01,  1.34470089e-02,  1.18324381e+00,  8.52215356e-01,
        8.42518612e-01,  1.46967141e-01,  2.21495314e-01,  1.02723789e-01,
       -1.19962861e+00,  1.57609374e+00,  1.40900595e-01,  2.15948951e-01,
        9.85673880e-02, -8.10893332e-02,  6.18496083e-01])
    #MAP = np.array([26.15362668, -0.32770818,  0.803825  ,  0.71214722,  1.35978106,
    #    0.15720997,  0.18496215,  0.04190095,  0.51648601,  0.12397726,
    #    0.05252478,  0.19255562,  0.06805631,  0.08109898,  0.58375491])

    for i_s, s_name in enumerate(short_names.keys()):
        #minar = np.zeros(15)
        #maxar= np.ones(15)
        #ave_val = np.zeros(15)
        #print(ave_val.tolist())
        #print(type(ave_val.tolist()))
        p = st.sidebar.slider(s_name, min_value= float(minar[i_s]), max_value= float(maxar[i_s]),
                              value=float(MAP[i_s]), step= 0.01)
        params.append(p)

    #get emu prediction
    Yemu_mean, Yemu_cov, time_emu = emu_predict(emu, params, inverse_tf_matrix, SS_mean)
    nplots = len(system_observables['Pb-Pb-2760'])
    #print(nplots)
    sns.set_context('poster')
    sns.set_style('whitegrid')
    fig, axs = plt.subplots(2,4,figsize=(25,15))
    axs = axs.flatten()
    last_obs = 0
    for i in range(0,nplots):
        ax = axs[i]
        cen_bins = obs_cent_list['Pb-Pb-2760'][system_observables['Pb-Pb-2760'][i]]
        #cen_bin_mid = np.array([(cen_pair[0]+cen_pair[1])/2 for cen_pair in cen_bins]).flatten()
        start_obs = last_obs
        end_obs = last_obs + len(cen_bins)
        y_values = Yemu_mean.flatten()[last_obs:end_obs]
        y_pred_err = np.sqrt(Yemu_cov.diagonal()[last_obs:end_obs])
        y_exp_values = y_exp[last_obs:end_obs]
        y_exp_err_values = np.sqrt(y_exp_variance[last_obs:end_obs])
        cen_bin_mid = np.arange(0,len(cen_bins))
        #print(cen_bins[0])
        #print(y_values)
        #print(y_pred_err)
        ax.errorbar(cen_bin_mid, y_exp_values, y_exp_err_values, color = 'red', fmt='o')
        ax.errorbar(cen_bin_mid, y_values, y_pred_err, color = 'blue')
        ax.set_title(system_observables['Pb-Pb-2760'][i])
        if i==1:
            cen_bin_mid = cen_bin_mid[0:-1:2]
            cen_bins = cen_bins[0:-1:2]
        ax.set_xticks(cen_bin_mid.flatten())
        ax.set_xticklabels(cen_bins, rotation =90)
        plt.tight_layout()
        last_obs = end_obs
    st.write(fig)
    #redraw plots
    #make_plot_altair(observables, Yemu_mean, Yemu_cov, y_exp, idf)
    #make_plot_eta_zeta(params)
    with st.expander("See how it works"):
        st.markdown('A description of the physics model and parameters can be found [here](https://indico.bnl.gov/event/6998/contributions/35770/attachments/27166/42261/JS_WS_2020_SIMS_v2.pdf).')
        st.markdown('The observables above (and additional ones not shown) are combined into [principal components](https://en.wikipedia.org/wiki/Principal_component_analysis) (PC).')
        st.markdown('A [Gaussian Process](https://en.wikipedia.org/wiki/Gaussian_process) (GP) is fitted to each of the dominant principal components by running our physics model on a coarse [space-filling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling) set of points in parameter space.')
        st.markdown('The Gaussian Process is then able to interpolate between these points, while estimating its own uncertainty.')
    st.markdown("How does simulation error compare with the experimental error?")
    chk = st.selectbox('Check it for',['20 observables with biggest error', 'All observables'])
    if st.button('Play video'):
        if chk == '20 observables with biggest error':
            st.image('animation.gif')
        else:
            st.image('animation_all.gif')
    #st.markdown('To update the widget with latest changes, click the button below, and then refresh your webpage.')
    #if st.button('(Update widget)'):
    #    subprocess.run("git pull origin master", shell=True)s


# %%
if __name__ == "__main__":
    main()
