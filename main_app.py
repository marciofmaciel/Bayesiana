import streamlit as st
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm, uniform, multivariate_normal
import time
import multiprocessing as mp
from functools import partial
import io
from fpdf import FPDF  # Adicione esta linha no topo do arquivo (instale com: pip install fpdf)
# --- 0. Configurações e Constantes Globais ---
st.set_page_config(layout="wide", page_title="Inferência Bayesiana para Compósitos via Ultrassom")
# Cores para gráficos
COLORS = sns.color_palette("viridis", 5)
# Propriedades padrão para Carbono-Epóxi Unidirecional (aproximadas, em GPa)
# C_ij em notação de Voigt (6x6)
# C11, C22, C33, C12, C13, C23, C44, C55, C66
DEFAULT_C_ORTHOTROPIC_GPA = np.array([
    [140.0, 5.0, 5.0, 0.0, 0.0, 0.0],
    [5.0, 10.0, 5.0, 0.0, 0.0, 0.0],
    [5.0, 5.0, 10.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 6.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 6.0]
])
# Ajustar para simetria C_ij = C_ji
DEFAULT_C_ORTHOTROPIC_GPA[0,1] = DEFAULT_C_ORTHOTROPIC_GPA[1,0] = 5.0
DEFAULT_C_ORTHOTROPIC_GPA[0,2] = DEFAULT_C_ORTHOTROPIC_GPA[2,0] = 5.0
DEFAULT_C_ORTHOTROPIC_GPA[1,2] = DEFAULT_C_ORTHOTROPIC_GPA[2,1] = 5.0
# Apenas os 9 independentes para ortotrópico
DEFAULT_C_PARAMS_GPA = {
    'C11': 140.0, 'C22': 10.0, 'C33': 10.0,
    'C12': 5.0, 'C13': 5.0, 'C23': 5.0,
    'C44': 3.0, 'C55': 6.0, 'C66': 6.0
}
DEFAULT_RHO_KG_M3 = 1550.0 # kg/m^3
# Parâmetros para MCMC
MCMC_DEFAULTS = {
    'n_iter': 10000,
    'burn_in': 2000,
    'n_chains': 2,
    'proposal_scale': 0.05 # Escala inicial para a covariância da proposta
}
# --- Funções Auxiliares ---
def voigt_to_full_tensor(C_voigt_gpa):
    """
    Converte a matriz de rigidez de Voigt (6x6) para o tensor completo (3x3x3x3).
    Assume C_voigt_gpa está em GPa. Retorna em Pa.
    """
    C_voigt = C_voigt_gpa * 1e9 # Convert GPa to Pa
    C_full = np.zeros((3, 3, 3, 3))
    # Mapeamento de Voigt
    voigt_map = [
        (0, 0), (1, 1), (2, 2),
        (1, 2), (0, 2), (0, 1)
    ]
    
    for i in range(6):
        for j in range(6):
            idx_i = voigt_map[i]
            idx_j = voigt_map[j]
            
            # Ajuste para os termos de cisalhamento (fator 2)
            factor_i = 1 if i < 3 else 2
            factor_j = 1 if j < 3 else 2
            
            C_full[idx_i[0], idx_i[1], idx_j[0], idx_j[1]] = C_voigt[i, j]
            
    # Para ortotrópico, precisamos preencher os termos simétricos corretamente
    # C_ijkl = C_jikl = C_ijlk = C_klij
    # A matriz de Voigt já deve ser simétrica.
    # A conversão direta já lida com a maioria.
    # Para os termos de cisalhamento, C_44 = C_2323, C_55 = C_1313, C_66 = C_1212
    # E suas permutações.
    
    # Para ortotrópico, a matriz de Voigt já é suficiente para o Christoffel.
    # A conversão para o tensor completo é mais complexa e geralmente não é feita
    # explicitamente para o Christoffel, que usa a forma contraída.
    # Vamos usar a forma contraída diretamente no ChristoffelSolver.
    return C_voigt # Retorna a matriz de Voigt em Pa

def get_orthotropic_C_voigt(params_gpa):
    """
    Constrói a matriz de rigidez 6x6 (Voigt) para um material ortotrópico
    a partir dos 9 parâmetros independentes.
    """
    C_voigt = np.zeros((6, 6))
    C_voigt[0, 0] = params_gpa['C11']
    C_voigt[1, 1] = params_gpa['C22']
    C_voigt[2, 2] = params_gpa['C33']
    C_voigt[0, 1] = C_voigt[1, 0] = params_gpa['C12']
    C_voigt[0, 2] = C_voigt[2, 0] = params_gpa['C13']
    C_voigt[1, 2] = C_voigt[2, 1] = params_gpa['C23']
    C_voigt[3, 3] = params_gpa['C44']
    C_voigt[4, 4] = params_gpa['C55']
    C_voigt[5, 5] = params_gpa['C66']
    return C_voigt * 1e9 # Retorna em Pa
def get_params_from_C_voigt(C_voigt_pa):
    """
    Extrai os 9 parâmetros independentes de uma matriz de rigidez 6x6 (Voigt)
    para um material ortotrópico. Retorna em GPa.
    """
    C_voigt_gpa = C_voigt_pa / 1e9
    params_gpa = {
        'C11': C_voigt_gpa[0, 0], 'C22': C_voigt_gpa[1, 1], 'C33': C_voigt_gpa[2, 2],
        'C12': C_voigt_gpa[0, 1], 'C13': C_voigt_gpa[0, 2], 'C23': C_voigt_gpa[1, 2],
        'C44': C_voigt_gpa[3, 3], 'C55': C_voigt_gpa[4, 4], 'C66': C_voigt_gpa[5, 5]
    }
    return params_gpa
def get_param_vector_from_dict(params_dict):
    """Converte o dicionário de parâmetros para um vetor numpy ordenado."""
    return np.array([
        params_dict['C11'], params_dict['C22'], params_dict['C33'],
        params_dict['C12'], params_dict['C13'], params_dict['C23'],
        params_dict['C44'], params_dict['C55'], params_dict['C66']
    ])
def get_param_dict_from_vector(param_vector):
    """Converte um vetor numpy ordenado para o dicionário de parâmetros."""
    return {
        'C11': param_vector[0], 'C22': param_vector[1], 'C33': param_vector[2],
        'C12': param_vector[3], 'C13': param_vector[4], 'C23': param_vector[5],
        'C44': param_vector[6], 'C55': param_vector[7], 'C66': param_vector[8]
    }
# --- 1. MÓDULO 1: Solver Christoffel Exato ---
class ChristoffelSolver:
    """
    Implementa o solver exato da equação de Christoffel para materiais anisotrópicos.
    Calcula as velocidades de fase e vetores de polarização para uma dada direção.
    """
    def __init__(self):
        pass

    def _get_christoffel_matrix(self, C_voigt_pa, n):
        """
        Calcula a matriz de Christoffel (3x3) para uma dada direção de propagação n.
        C_voigt_pa: Matriz de rigidez 6x6 em Pa.
        n: Vetor unitário de direção de propagação [n1, n2, n3].
        """
        n1, n2, n3 = n
        Gamma = np.zeros((3, 3))
        # Simplificado para ortotrópico
        Gamma[0,0] = C_voigt_pa[0,0]*n1**2 + C_voigt_pa[5,5]*n2**2 + C_voigt_pa[4,4]*n3**2
        Gamma[1,1] = C_voigt_pa[5,5]*n1**2 + C_voigt_pa[1,1]*n2**2 + C_voigt_pa[3,3]*n3**2
        Gamma[2,2] = C_voigt_pa[4,4]*n1**2 + C_voigt_pa[3,3]*n2**2 + C_voigt_pa[2,2]*n3**2
        Gamma[0,1] = Gamma[1,0] = (C_voigt_pa[0,1] + C_voigt_pa[5,5])*n1*n2
        Gamma[0,2] = Gamma[2,0] = (C_voigt_pa[0,2] + C_voigt_pa[4,4])*n1*n3
        Gamma[1,2] = Gamma[2,1] = (C_voigt_pa[1,2] + C_voigt_pa[3,3])*n2*n3
        return Gamma

    def solve(self, C_voigt_pa, rho, n):
        """
        Resolve a equação de Christoffel para as velocidades de fase e polarizações.
        C_voigt_pa: Matriz de rigidez 6x6 em Pa.
        rho: Densidade em kg/m^3.
        n: Vetor unitário de direção de propagação [n1, n2, n3].
        Retorna: (velocidades_fase, vetores_polarizacao, modos_identificados)
        """
        if not np.isclose(np.linalg.norm(n), 1.0):
            raise ValueError("Vetor de direção n deve ser unitário.")
        Gamma = self._get_christoffel_matrix(C_voigt_pa, n)
        eigenvalues, eigenvectors = la.eigh(Gamma)
        eigenvalues[eigenvalues < 0] = 0
        v_squared = eigenvalues / rho
        velocities = np.sqrt(v_squared)
        dot_products = np.abs(np.dot(eigenvectors.T, n))
        qP_idx = np.argmax(dot_products)
        qS_indices = [i for i in range(3) if i != qP_idx]
        ordered_indices = [qP_idx] + qS_indices
        ordered_velocities = velocities[ordered_indices]
        ordered_eigenvectors = eigenvectors[:, ordered_indices]
        modes = ["qP", "qS1", "qS2"]
        return ordered_velocities, ordered_eigenvectors, modes

# --- 2. MÓDULO 2: Modelo Ultrassônico Realista ---
class UltrasonicModel:
   #Simula o comportamento de ondas ultrassônicas em um material compósito,incluindo TOF, atenuação e uma representação simplificada de dispersão.
    
    def __init__(self, christoffel_solver):
        self.christoffel_solver = christoffel_solver

    def predict_tof(self, C_voigt_pa, rho, thickness_m, n, mode_idx=0):
        """
        Prevê o Tempo de Voo (TOF) para um modo específico.
        mode_idx: 0 para qP, 1 para qS1, 2 para qS2.
        """
        try:
            velocities, _, _ = self.christoffel_solver.solve(C_voigt_pa, rho, n)
            phase_velocity = velocities[mode_idx]
            if phase_velocity <= 0:
                return np.inf # Velocidade não física
            tof = thickness_m / phase_velocity
            return tof
        except ValueError: # Erro do solver Christoffel
            return np.inf
        except IndexError: # mode_idx inválido
            return np.inf

    def predict_attenuation(self, C_voigt_pa, rho, n, frequency_hz, thickness_m, 
                            mode_idx=0, loss_factor_base=0.01, loss_factor_freq_exp=1.0):
        """
        Prevê a atenuação em dB para um modo específico.
        Modelo simplificado de atenuação viscoelástica: alpha = (pi * f * loss_factor) / v
        loss_factor_base: Fator de perda base (adimensional).
        loss_factor_freq_exp: Expoente da dependência da frequência (1.0 para linear).
        """
        try:
            velocities, _, _ = self.christoffel_solver.solve(C_voigt_pa, rho, n)
            phase_velocity = velocities[mode_idx]
            if phase_velocity <= 0:
                return np.inf
            
            # Fator de perda pode depender da direção e do modo, mas aqui simplificamos
            # para um valor base e dependência da frequência.
            loss_factor = loss_factor_base * (frequency_hz / 1e6)**loss_factor_freq_exp # Normaliza freq para MHz
            
            alpha_np = (np.pi * frequency_hz * loss_factor) / phase_velocity # Coeficiente de atenuação neperiano (Np/m)
            
            attenuation_db = 20 * np.log10(np.exp(alpha_np * thickness_m)) # Atenuação em dB
            return attenuation_db
        except ValueError:
            return np.inf
        except IndexError:
            return np.inf

    def simulate_signal(self, C_voigt_pa, rho, thickness_m, n, frequency_hz, 
                        mode_idx=0, noise_level=0.05, loss_factor_base=0.01, 
                        loss_factor_freq_exp=1.0, dt=1e-8, num_points=1000):
        """
        Simula um sinal ultrassônico de pulso-eco (simplificado).
        Gera um pulso Ricker, aplica TOF, atenuação e adiciona ruído.
        """
        tof = self.predict_tof(C_voigt_pa, rho, thickness_m, n, mode_idx)
        attenuation_db = self.predict_attenuation(C_voigt_pa, rho, n, frequency_hz, 
thickness_m, mode_idx, loss_factor_base,
loss_factor_freq_exp)
        if tof == np.inf or attenuation_db == np.inf:
            return np.linspace(0, num_points * dt, num_points), np.zeros(num_points)

        # Pulso Ricker (derivada segunda de Gaussiana)
        t = np.linspace(-5/frequency_hz, 5/frequency_hz, num_points)
        ricker_pulse = (1 - 2 * (np.pi * frequency_hz * t)**2) * np.exp(-(np.pi * frequency_hz * t)**2)
        
        # Escala de tempo para o sinal completo
        time_axis = np.linspace(0, num_points * dt, num_points)
        
        # Atraso do pulso
        shifted_pulse = np.interp(time_axis - tof, t, ricker_pulse, left=0, right=0)
        
        # Aplica atenuação (converte dB para fator linear)
        attenuation_factor = 10**(-attenuation_db / 20)
        attenuated_pulse = shifted_pulse * attenuation_factor
        
        # Adiciona ruído Gaussiano
        noise = np.random.normal(0, noise_level * np.max(np.abs(attenuated_pulse)), num_points)
        final_signal = attenuated_pulse + noise
        
        return time_axis, final_signal

# --- 3. MÓDULO 3: Likelihood Bayesiana Precisa ---
class BayesianLikelihood:
    """
    Calcula a função de verossimilhança (likelihood) para o modelo Bayesiano,
    assumindo erros Gaussianos.
    """
    def __init__(self, ultrasonic_model: UltrasonicModel):
        self.ultrasonic_model = ultrasonic_model

    def log_likelihood(self, params_gpa, rho, experimental_data):
        """
        Calcula o log da verossimilhança para um conjunto de parâmetros (C_ij, rho).
        experimental_data: DataFrame com colunas 'direction_n', 'mode_idx', 'v_exp', 'sigma_exp'.
        """
        C_voigt_pa = get_orthotropic_C_voigt(params_gpa)
        
        log_like = 0.0
        for _, row in experimental_data.iterrows():
            n = np.array(row['direction_n'])
            mode_idx = row['mode_idx']
            v_exp = row['v_exp']
            sigma_exp = row['sigma_exp']
            thickness_m = row['thickness_m'] # Assumindo thickness_m está no DataFrame
            
            # Prever TOF e converter para velocidade
            tof_pred = self.ultrasonic_model.predict_tof(C_voigt_pa, rho, thickness_m, n, mode_idx)
            if tof_pred == np.inf or tof_pred <= 0:
                return -np.inf # Parâmetros não físicos
            v_pred = thickness_m / tof_pred # Velocidade prevista
            
            # Termo da likelihood Gaussiana
            log_like += norm.logpdf(v_exp, loc=v_pred, scale=sigma_exp)
            
        return log_like

    def log_prior(self, params_gpa, rho, prior_bounds_gpa, rho_prior_bounds):
        """
        Calcula o log do prior para um conjunto de parâmetros.
        Assume priors uniformes para C_ij e rho.
        """
        # Prior para C_ij
        for param_name, value in params_gpa.items():
            if not (prior_bounds_gpa[param_name][0] <= value <= prior_bounds_gpa[param_name][1]):
                return -np.inf
        
        # Prior para rho
        if not (rho_prior_bounds[0] <= rho <= rho_prior_bounds[1]):
            return -np.inf
            
        # Priors uniformes têm densidade constante dentro dos limites,
        # então o log da densidade é constante (e pode ser ignorado para MCMC)
        # ou log(1/range). Aqui, retornamos 0 se dentro dos limites.
        return 0.0

    def log_posterior(self, params_gpa, rho, experimental_data, prior_bounds_gpa, rho_prior_bounds):
        """
        Calcula o log da posterior (log_likelihood + log_prior).
        """
        lp = self.log_prior(params_gpa, rho, prior_bounds_gpa, rho_prior_bounds)
        if lp == -np.inf:
            return -np.inf
        
        ll = self.log_likelihood(params_gpa, rho, experimental_data)
        
        return lp + ll

# --- 4. MÓDULO 4: MCMC Metropolis-Hastings ---
class MCMCSampler:
    """
Implementa o algoritmo Metropolis-Hastings para amostragem da distribuição posterior.
Inclui adaptação da covariância da proposta e diagnósticos de convergência.
    """
    def __init__(self, bayesian_likelihood: BayesianLikelihood):
        self.bayesian_likelihood = bayesian_likelihood

    def _run_chain(self, chain_id, initial_state, n_iter, burn_in, proposal_cov_initial, 
                   experimental_data, prior_bounds_gpa, rho_prior_bounds, adapt_interval=100):
        """
        Função para rodar uma única cadeia MCMC.
        Retorna amostras, log_posteriors e taxa de aceitação.
        """
        n_params = len(initial_state) - 1 # C_params + rho
        
        samples = np.zeros((n_iter, n_params + 1))
        log_posteriors = np.zeros(n_iter)
        
        current_params_gpa = get_param_dict_from_vector(initial_state[:-1])
        current_rho = initial_state[-1]
        
        current_log_post = self.bayesian_likelihood.log_posterior(
            current_params_gpa, current_rho, experimental_data, prior_bounds_gpa, rho_prior_bounds
        )
        
        accepted_count = 0
        
        # Adaptação da covariância da proposta
        proposal_cov = proposal_cov_initial.copy()
        
        for i in range(n_iter):
            # Propor novo estado
            proposed_state = multivariate_normal.rvs(mean=initial_state, cov=proposal_cov)
            proposed_params_gpa = get_param_dict_from_vector(proposed_state[:-1])
            proposed_rho = proposed_state[-1]
            
            proposed_log_post = self.bayesian_likelihood.log_posterior(
                proposed_params_gpa, proposed_rho, experimental_data, prior_bounds_gpa, rho_prior_bounds
            )
            
            # Calcular razão de aceitação (em escala logarítmica)
            log_alpha = proposed_log_post - current_log_post
            alpha = min(1.0, np.exp(log_alpha))
            
            # Aceitar ou rejeitar
            if np.random.rand() < alpha:
                current_params_gpa = proposed_params_gpa
                current_rho = proposed_rho
                current_log_post = proposed_log_post
                initial_state = proposed_state # Atualiza para próxima proposta
                accepted_count += 1
            
            samples[i, :-1] = get_param_vector_from_dict(current_params_gpa)
            samples[i, -1] = current_rho
            log_posteriors[i] = current_log_post

            # Adaptação da covariância da proposta (durante burn-in)
            if i < burn_in and (i + 1) % adapt_interval == 0:
                current_acceptance_rate = accepted_count / (i + 1)
                if current_acceptance_rate < 0.2: # Muito baixa, reduzir passo
                    proposal_cov *= 0.8
                elif current_acceptance_rate > 0.5: # Muito alta, aumentar passo
                    proposal_cov *= 1.2
                # Reset accepted_count for next adaptation interval
                accepted_count = 0 
        
        acceptance_rate = accepted_count / n_iter
        return samples, log_posteriors, acceptance_rate

    def sample(self, initial_states, n_iter, burn_in, proposal_scale, 
               experimental_data, prior_bounds_gpa, rho_prior_bounds):
        """
        Roda múltiplas cadeias MCMC em sequência (sem multiprocessing).
        Isso evita problemas de pickling com classes definidas em __main__.
        """
        n_chains = len(initial_states)
        n_params = len(initial_states[0]) # C_params + rho

        # Covariância inicial da proposta (diagonal)
        proposal_cov_initial = np.eye(n_params) * proposal_scale**2

        st.write(f"Iniciando {n_chains} cadeias MCMC com {n_iter} iterações cada (burn-in: {burn_in}).")

        all_samples = []
        all_log_posteriors = []
        all_acceptance_rates = []

        for i in range(n_chains):
            samples, log_posteriors, acceptance_rate = self._run_chain(
                i, initial_states[i], n_iter, burn_in, proposal_cov_initial,
                experimental_data, prior_bounds_gpa, rho_prior_bounds
            )
            all_samples.append(samples)
            all_log_posteriors.append(log_posteriors)
            all_acceptance_rates.append(acceptance_rate)

        return all_samples, all_log_posteriors, all_acceptance_rates

    def calculate_rhat(self, all_samples, burn_in):
        """
        Calcula a estatística R-hat de Gelman-Rubin para cada parâmetro.
        """
        if len(all_samples) < 2:
            return None # R-hat requer pelo menos 2 cadeias
        
        n_chains = len(all_samples)
        n_params = all_samples[0].shape[1]
        
        rhat_values = np.zeros(n_params)
        
        for p in range(n_params):
            chain_samples = np.array([s[burn_in:, p] for s in all_samples])
            
            # Variância dentro da cadeia (W)
            W = np.mean(np.var(chain_samples, axis=1, ddof=1))
            
            # Variância entre cadeias (B)
            chain_means = np.mean(chain_samples, axis=1)
            B = len(chain_samples[0]) * np.var(chain_means, ddof=1)
            
            # Estimativa da variância posterior (V_hat)
            V_hat = (len(chain_samples[0]) - 1) / len(chain_samples[0]) * W + (1 / len(chain_samples[0])) * B
            
            rhat_values[p] = np.sqrt(V_hat / W) if W > 0 else np.nan # Evitar divisão por zero
            
        return rhat_values

    def calculate_ess(self, all_samples, burn_in):
        """
        Calcula o Effective Sample Size (ESS) para cada parâmetro.
        Simplificado: usa a autocorrelação de uma cadeia representativa.
        Para uma implementação mais robusta, seria necessário um pacote como ArviZ.
        """
        if len(all_samples) == 0:
            return None
        
        # Usar a primeira cadeia para estimar autocorrelação
        representative_chain = all_samples[0][burn_in:, :]
        n_params = representative_chain.shape[1]
        
        ess_values = np.zeros(n_params)
        
        for p in range(n_params):
            samples_p = representative_chain[:, p]
            
            # Estimar autocorrelação (simplificado)
            # Usar fft para calcular autocorrelação
            n_eff = len(samples_p)
            f = np.fft.fft(samples_p - np.mean(samples_p), n=2*n_eff)
            acf = np.fft.ifft(f * np.conjugate(f))[:n_eff].real
            acf /= acf[0] # Normalizar
            
            # Sum of positive autocorrelations
            sum_rho_k = 0
            for k in range(1, n_eff):
                if acf[k] > 0:
                    sum_rho_k += acf[k]
                else:
                    break # Stop when autocorrelation becomes negative
            
            ess_values[p] = n_eff / (1 + 2 * sum_rho_k)
            
        return ess_values

# --- 5. MÓDULO 5: Validação Completa ---
class ValidationTools:
    """
Fornece ferramentas para análise de sensibilidade, identificabilidade e validação
do modelo Bayesiano.
    """
    def __init__(self, bayesian_likelihood: BayesianLikelihood):
        self.bayesian_likelihood = bayesian_likelihood
    def sensitivity_to_prior(self, posterior_samples, prior_bounds_gpa, rho_prior_bounds):
        """
        Compara a dispersão posterior com a dispersão do prior para avaliar a informatividade dos dados.
        Retorna um DataFrame com SD_prior, SD_posterior e a razão.
        """
        param_names = list(prior_bounds_gpa.keys()) + ['rho']
        
        sd_prior = []
        sd_posterior = np.std(posterior_samples, axis=0)
        
        for name in prior_bounds_gpa.keys():
            lower, upper = prior_bounds_gpa[name]
            sd_prior.append(uniform.std(loc=lower, scale=upper-lower))
        
        lower_rho, upper_rho = rho_prior_bounds
        sd_prior.append(uniform.std(loc=lower_rho, scale=upper_rho-lower_rho))
        
        sd_prior = np.array(sd_prior)
        
        results = pd.DataFrame({
            'Parâmetro': param_names,
            'SD Prior': sd_prior,
            'SD Posterior': sd_posterior,
            'Razão (SD Prior/SD Post)': sd_prior / sd_posterior
        })
        return results

    def identifiability_analysis(self, posterior_samples):
        """
        Calcula a matriz de correlação posterior para identificar parâmetros correlacionados.
        """
        correlation_matrix = np.corrcoef(posterior_samples, rowvar=False)
        return correlation_matrix

    def posterior_predictive_check(self, posterior_samples, experimental_data, n_simulations=100):
        """
        Realiza um Posterior Predictive Check (PPC).
        Simula dados com base nas amostras posteriores e compara com os dados experimentais.
        Retorna um DataFrame com v_exp, v_pred_mean, v_pred_std e um p-value Bayesiano.
        """
        n_exp_points = len(experimental_data)
        simulated_velocities = np.zeros((n_simulations, n_exp_points))
        
        # Amostrar aleatoriamente do posterior para simular
        sample_indices = np.random.choice(posterior_samples.shape[0], n_simulations, replace=False)
        
        for i, idx in enumerate(sample_indices):
            params_gpa = get_param_dict_from_vector(posterior_samples[idx, :-1])
            rho = posterior_samples[idx, -1]
            C_voigt_pa = get_orthotropic_C_voigt(params_gpa)
            
            for j, (_, row) in enumerate(experimental_data.iterrows()):
                n = np.array(row['direction_n'])
                mode_idx = row['mode_idx']
                thickness_m = row['thickness_m']
                
                tof_pred = self.bayesian_likelihood.ultrasonic_model.predict_tof(C_voigt_pa, rho, thickness_m, n, mode_idx)
                if tof_pred == np.inf or tof_pred <= 0:
                    simulated_velocities[i, j] = np.nan
                else:
                    simulated_velocities[i, j] = thickness_m / tof_pred
        
        # Calcular estatísticas das velocidades simuladas
        v_pred_mean = np.nanmean(simulated_velocities, axis=0)
        v_pred_std = np.nanstd(simulated_velocities, axis=0)
        
        # Calcular p-value Bayesiano (simplificado: proporção de vezes que a estatística simulada é maior que a observada)
        # Usamos a estatística chi-quadrado como exemplo
        chi2_obs = np.sum(((experimental_data['v_exp'].values - v_pred_mean) / experimental_data['sigma_exp'].values)**2)
        
        chi2_sim = np.zeros(n_simulations)
        for i in range(n_simulations):
            # Para cada simulação, calculamos o chi2 em relação à média preditiva
            # Uma forma mais robusta seria comparar com os próprios dados simulados
            chi2_sim[i] = np.sum(((simulated_velocities[i, :] - v_pred_mean) / experimental_data['sigma_exp'].values)**2)
        
        p_value_bayesian = np.mean(chi2_sim > chi2_obs)
        
        results = experimental_data.copy()
        results['v_pred_mean'] = v_pred_mean
        results['v_pred_std'] = v_pred_std
        
        return results, p_value_bayesian

    def loo_cv_conceptual(self):
        """
        Apresenta uma explicação conceitual do Leave-One-Out Cross-Validation (LOO-CV),
        já que a implementação completa é complexa e requer bibliotecas como ArviZ.
        """
        st.subheader("Leave-One-Out Cross-Validation (LOO-CV)")
        st.write("""
        O LOO-CV é uma técnica de validação cruzada que avalia a capacidade preditiva do modelo.
        Para cada ponto de dado experimental, o modelo é treinado com *todos os outros* pontos
        e, em seguida, a probabilidade de prever o ponto de dado "deixado de fora" é calculada.
        
        **Como funciona:**
        1.  Para cada medição $v_i$ no conjunto de dados:
            *   Treine o modelo Bayesiano usando todos os dados *exceto* $v_i$.
            *   Use o modelo treinado para prever a distribuição de $v_i$.
            *   Calcule a probabilidade logarítmica preditiva de $v_i$ dado o modelo treinado com os dados restantes.
        2.  A soma dessas probabilidades logarítmicas preditivas (ELPD_LOO) é uma medida da capacidade
            preditiva geral do modelo.
        
        **Benefícios:**
        *   Fornece uma estimativa robusta da capacidade de generalização do modelo.
        *   Ajuda a identificar pontos de dados influentes ou outliers.
        *   Útil para comparação de modelos (modelos com ELPD_LOO maior são preferíveis).
        
        **Desafios:**
        *   Computacionalmente intensivo, pois requer rodar a inferência $N$ vezes (onde $N$ é o número de dados).
        *   Métodos eficientes como PSIS-LOO (Pareto Smoothed Importance Sampling) são usados na prática
            (implementados em bibliotecas como ArviZ) para evitar o re-treinamento completo.
        
        **Interpretação:**
        *   Um ELPD_LOO mais alto indica um modelo com melhor poder preditivo.
        *   Valores de "k-hat" (diagnóstico de PSIS-LOO) acima de 0.7 podem indicar pontos de dados
            problemáticos ou um modelo mal-especificado.
        """)

#--- Interface Streamlit ---
st.sidebar.title("Navegação")
page = st.sidebar.radio("Selecione um Módulo", [
    "Módulo 1: Christoffel Solver",
    "Módulo 2: Modelo Ultrassônico",
    "Módulo 3: Likelihood Bayesiana",
    "Módulo 4: MCMC Metropolis-Hastings",
    "Módulo 5: Validação Completa",
    "Módulo 6: Módulos Elásticos"
])

# Inicializar solvers e modelos (singleton pattern para evitar recriação)
if 'christoffel_solver' not in st.session_state:
    st.session_state.christoffel_solver = ChristoffelSolver()
if 'ultrasonic_model' not in st.session_state:
    st.session_state.ultrasonic_model = UltrasonicModel(st.session_state.christoffel_solver)
if 'bayesian_likelihood' not in st.session_state:
    st.session_state.bayesian_likelihood = BayesianLikelihood(st.session_state.ultrasonic_model)
if 'mcmc_sampler' not in st.session_state:
    st.session_state.mcmc_sampler = MCMCSampler(st.session_state.bayesian_likelihood)
if 'validation_tools' not in st.session_state:
    st.session_state.validation_tools = ValidationTools(st.session_state.bayesian_likelihood)

# --- Módulo 1: Christoffel Solver ---
if page == "Módulo 1: Christoffel Solver":
    st.title("Módulo 1: Christoffel Solver Exato")
    st.write("""
    Este módulo implementa o solver exato da equação de Christoffel para materiais anisotrópicos.
    Ele calcula as velocidades de fase e os vetores de polarização para uma dada direção de propagação
    e propriedades elásticas do material.
    """)
    st.subheader("Propriedades do Material (Carbono-Epóxi Padrão)")
    
    # Usar st.session_state para persistir os parâmetros C
    if 'c_params_gpa' not in st.session_state:
        st.session_state.c_params_gpa = DEFAULT_C_PARAMS_GPA.copy()
    if 'rho_kg_m3' not in st.session_state:
        st.session_state.rho_kg_m3 = DEFAULT_RHO_KG_M3

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### Rigidez Axial (GPa)")
        st.session_state.c_params_gpa['C11'] = st.number_input("C11 (GPa)", value=st.session_state.c_params_gpa['C11'], min_value=1.0, max_value=300.0, step=1.0)
        st.session_state.c_params_gpa['C22'] = st.number_input("C22 (GPa)", value=st.session_state.c_params_gpa['C22'], min_value=1.0, max_value=300.0, step=1.0)
        st.session_state.c_params_gpa['C33'] = st.number_input("C33 (GPa)", value=st.session_state.c_params_gpa['C33'], min_value=1.0, max_value=300.0, step=1.0)
    with col2:
        st.markdown("##### Acoplamento (GPa)")
        st.session_state.c_params_gpa['C12'] = st.number_input("C12 (GPa)", value=st.session_state.c_params_gpa['C12'], min_value=0.0, max_value=100.0, step=0.1)
        st.session_state.c_params_gpa['C13'] = st.number_input("C13 (GPa)", value=st.session_state.c_params_gpa['C13'], min_value=0.0, max_value=100.0, step=0.1)
        st.session_state.c_params_gpa['C23'] = st.number_input("C23 (GPa)", value=st.session_state.c_params_gpa['C23'], min_value=0.0, max_value=100.0, step=0.1)
    with col3:
        st.markdown("##### Cisalhamento (GPa)")
        st.session_state.c_params_gpa['C44'] = st.number_input("C44 (GPa)", value=st.session_state.c_params_gpa['C44'], min_value=0.1, max_value=100.0, step=0.1)
        st.session_state.c_params_gpa['C55'] = st.number_input("C55 (GPa)", value=st.session_state.c_params_gpa['C55'], min_value=0.1, max_value=100.0, step=0.1)
        st.session_state.c_params_gpa['C66'] = st.number_input("C66 (GPa)", value=st.session_state.c_params_gpa['C66'], min_value=0.1, max_value=100.0, step=0.1)
        st.session_state.rho_kg_m3 = st.number_input("Densidade ρ (kg/m³)", value=st.session_state.rho_kg_m3, min_value=500.0, max_value=3000.0, step=1.0)

    st.subheader("Direção de Propagação")
    theta_deg = st.slider("Ângulo Polar θ (graus, do eixo Z)", 0, 180, 0)
    phi_deg = st.slider("Ângulo Azimutal φ (graus, no plano XY)", 0, 360, 0)

    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    n = np.array([
        np.sin(theta_rad) * np.cos(phi_rad),
        np.sin(theta_rad) * np.sin(phi_rad),
        np.cos(theta_rad)
    ])
    st.write(f"Vetor de direção n: [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]")

    if st.button("Calcular Velocidades"):
        C_voigt_pa = get_orthotropic_C_voigt(st.session_state.c_params_gpa)
        try:
            velocities, eigenvectors, modes = st.session_state.christoffel_solver.solve(C_voigt_pa, st.session_state.rho_kg_m3, n)
            st.success("Cálculo Concluído!")
            
            st.subheader("Resultados")
            for i in range(3):
                st.write(f"**Modo {modes[i]}**")
                st.write(f"  Velocidade de Fase: {velocities[i]:.2f} m/s")
                st.write(f"  Vetor de Polarização: [{eigenvectors[0, i]:.3f}, {eigenvectors[1, i]:.3f}, {eigenvectors[2, i]:.3f}]")
            
            st.markdown("---")
            st.subheader("Matriz de Rigidez (Voigt, GPa)")
            st.dataframe(pd.DataFrame(C_voigt_pa / 1e9, columns=[f'C{i+1}' for i in range(6)], index=[f'C{i+1}' for i in range(6)]))
            st.subheader("Matriz de Christoffel (Γ)")
            Gamma = st.session_state.christoffel_solver._get_christoffel_matrix(C_voigt_pa, n)
            st.dataframe(pd.DataFrame(Gamma))

        except ValueError as e:
            st.error(f"Erro no cálculo: {e}")
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado: {e}")

#--- Módulo 2: Modelo Ultrassônico Realista ---
elif page == "Módulo 2: Modelo Ultrassônico":
    st.title("Módulo 2: Modelo Ultrassônico Realista")
    st.write("""
    Este módulo simula o comportamento de um sinal ultrassônico através de um material compósito,
    considerando o Tempo de Voo (TOF), atenuação e ruído.
    """)
    st.subheader("Parâmetros do Ensaio Ultrassônico")
    col1, col2 = st.columns(2)
    with col1:
        thickness_mm = st.number_input("Espessura da Amostra (mm)", value=5.0, min_value=0.1, max_value=50.0, step=0.1)
        frequency_mhz = st.number_input("Frequência do Transdutor (MHz)", value=5.0, min_value=0.1, max_value=20.0, step=0.1)
        mode_selection = st.selectbox("Modo de Onda", ["qP (Quasi-Longitudinal)", "qS1 (Quasi-Cisalhante 1)", "qS2 (Quasi-Cisalhante 2)"])
        mode_idx = {"qP (Quasi-Longitudinal)": 0, "qS1 (Quasi-Cisalhante 1)": 1, "qS2 (Quasi-Cisalhante 2)": 2}[mode_selection]
    with col2:
        noise_level = st.slider("Nível de Ruído (0-1)", value=0.05, min_value=0.0, max_value=1.0, step=0.01)
        loss_factor_base = st.number_input("Fator de Perda Base", value=0.01, min_value=0.001, max_value=0.1, step=0.001, format="%.3f")
        loss_factor_freq_exp = st.number_input("Expoente da Frequência para Perda", value=1.0, min_value=0.0, max_value=2.0, step=0.1)
    
    st.subheader("Direção de Propagação")
    theta_deg_ult = st.slider("Ângulo Polar θ (graus)", 0, 180, 0, key="theta_ult")
    phi_deg_ult = st.slider("Ângulo Azimutal φ (graus)", 0, 360, 0, key="phi_ult")

    theta_rad_ult = np.deg2rad(theta_deg_ult)
    phi_rad_ult = np.deg2rad(phi_deg_ult)
    n_ult = np.array([
        np.sin(theta_rad_ult) * np.cos(phi_rad_ult),
        np.sin(theta_rad_ult) * np.sin(phi_rad_ult),
        np.cos(theta_rad_ult)
    ])

    if st.button("Simular Sinal Ultrassônico"):
        C_voigt_pa = get_orthotropic_C_voigt(st.session_state.c_params_gpa)
        rho = st.session_state.rho_kg_m3
        thickness_m = thickness_mm / 1000.0
        frequency_hz = frequency_mhz * 1e6

        try:
            velocities, _, _ = st.session_state.christoffel_solver.solve(C_voigt_pa, rho, n_ult)
            phase_velocity = velocities[mode_idx]
            
            tof = st.session_state.ultrasonic_model.predict_tof(C_voigt_pa, rho, thickness_m, n_ult, mode_idx)
            attenuation_db = st.session_state.ultrasonic_model.predict_attenuation(C_voigt_pa, rho, n_ult, frequency_hz, 
thickness_m, mode_idx, loss_factor_base,
loss_factor_freq_exp)
            st.subheader("Resultados da Simulação")
            st.write(f"Velocidade de Fase ({mode_selection}): {phase_velocity:.2f} m/s")
            st.write(f"Tempo de Voo (TOF): {tof * 1e6:.2f} µs")
            st.write(f"Atenuação: {attenuation_db:.2f} dB")

            time_axis, signal = st.session_state.ultrasonic_model.simulate_signal(
                C_voigt_pa, rho, thickness_m, n_ult, frequency_hz, mode_idx, 
                noise_level, loss_factor_base, loss_factor_freq_exp
            )

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_axis * 1e6, signal)
            ax.set_title("Sinal Ultrassônico Simulado")
            ax.set_xlabel("Tempo (µs)")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
            st.pyplot(fig)

        except ValueError as e:
            st.error(f"Erro no cálculo: {e}")
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado: {e}")

#--- Módulo 3: Likelihood Bayesiana Precisa ---
elif page == "Módulo 3: Likelihood Bayesiana":
    st.title("Módulo 3: Likelihood Bayesiana Precisa")
    st.write("""
    Este módulo calcula o log da verossimilhança (likelihood) e do prior para um conjunto de parâmetros
    (constantes elásticas e densidade) dado um conjunto de dados experimentais de velocidade.
    """)
    st.subheader("Dados Experimentais (Velocidades)")
    st.write("Insira dados de velocidade experimental (m/s), desvio padrão (m/s), direção (θ, φ) e espessura (mm).")

    # Estrutura para dados experimentais
    if 'experimental_data' not in st.session_state:
        st.session_state.experimental_data = pd.DataFrame({
            'direction_n': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.707, 0.0, 0.707]],
            'mode_idx': [0, 0, 0, 0], # qP
            'v_exp': [7400.0, 2100.0, 2000.0, 4500.0],
            'sigma_exp': [50.0, 50.0, 50.0, 70.0],
            'thickness_m': [5.0/1000, 5.0/1000, 5.0/1000, 5.0/1000]
        })
    
    edited_df = st.data_editor(st.session_state.experimental_data, num_rows="dynamic", key="exp_data_editor")
    st.session_state.experimental_data = edited_df

    st.subheader("Priors (Limites Uniformes para C_ij em GPa, ρ em kg/m³)")
    if 'prior_bounds_gpa' not in st.session_state:
        st.session_state.prior_bounds_gpa = {
            'C11': [100.0, 180.0], 'C22': [5.0, 15.0], 'C33': [5.0, 15.0],
            'C12': [2.0, 8.0], 'C13': [2.0, 8.0], 'C23': [2.0, 8.0],
            'C44': [1.0, 5.0], 'C55': [3.0, 10.0], 'C66': [3.0, 10.0]
        }
    if 'rho_prior_bounds' not in st.session_state:
        st.session_state.rho_prior_bounds = [1400.0, 1700.0]

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown("##### Rigidez Axial")
        for p in ['C11', 'C22', 'C33']:
            st.session_state.prior_bounds_gpa[p] = st.slider(f"{p} (GPa)", 0.0, 300.0, st.session_state.prior_bounds_gpa[p], key=f"prior_{p}")
    with col_p2:
        st.markdown("##### Acoplamento")
        for p in ['C12', 'C13', 'C23']:
            st.session_state.prior_bounds_gpa[p] = st.slider(f"{p} (GPa)", 0.0, 100.0, st.session_state.prior_bounds_gpa[p], key=f"prior_{p}")
    with col_p3:
        st.markdown("##### Cisalhamento")
        for p in ['C44', 'C55', 'C66']:
            st.session_state.prior_bounds_gpa[p] = st.slider(f"{p} (GPa)", 0.0, 100.0, st.session_state.prior_bounds_gpa[p], key=f"prior_{p}")
        st.session_state.rho_prior_bounds = st.slider("Densidade ρ (kg/m³)", 500.0, 3000.0, st.session_state.rho_prior_bounds, key="prior_rho")

    st.subheader("Avaliar Log-Likelihood/Posterior para Parâmetros Atuais")
    if st.button("Calcular Log-Likelihood e Log-Posterior"):
        try:
            ll = st.session_state.bayesian_likelihood.log_likelihood(
                st.session_state.c_params_gpa, st.session_state.rho_kg_m3, st.session_state.experimental_data
            )
            lp = st.session_state.bayesian_likelihood.log_prior(
                st.session_state.c_params_gpa, st.session_state.rho_kg_m3, st.session_state.prior_bounds_gpa, st.session_state.rho_prior_bounds
            )
            lpost = st.session_state.bayesian_likelihood.log_posterior(
                st.session_state.c_params_gpa, st.session_state.rho_kg_m3, st.session_state.experimental_data, 
                st.session_state.prior_bounds_gpa, st.session_state.rho_prior_bounds
            )
            st.success("Cálculo Concluído!")
            st.write(f"Log-Likelihood: {ll:.2f}")
            st.write(f"Log-Prior: {lp:.2f}")
            st.write(f"Log-Posterior: {lpost:.2f}")
        except Exception as e:
            st.error(f"Erro no cálculo: {e}")

#--- Módulo 4: MCMC Metropolis-Hastings ---
elif page == "Módulo 4: MCMC Metropolis-Hastings":
    st.title("Módulo 4: MCMC Metropolis-Hastings")
    st.write("""
    Este módulo implementa o algoritmo Metropolis-Hastings para amostrar a distribuição posterior
    dos parâmetros do material. Inclui adaptação da covariância da proposta e diagnósticos de convergência.
    """)
    st.subheader("Configurações MCMC")
    col_mcmc1, col_mcmc2 = st.columns(2)
    with col_mcmc1:
        n_iter = st.number_input("Número de Iterações por Cadeia", value=MCMC_DEFAULTS['n_iter'], min_value=1000, max_value=100000, step=1000)
        burn_in = st.number_input("Iterações de Burn-in", value=MCMC_DEFAULTS['burn_in'], min_value=100, max_value=n_iter // 2, step=100)
    with col_mcmc2:
        n_chains = st.number_input("Número de Cadeias", value=MCMC_DEFAULTS['n_chains'], min_value=1, max_value=4, step=1)
        proposal_scale = st.number_input("Escala Inicial da Proposta", value=MCMC_DEFAULTS['proposal_scale'], min_value=0.001, max_value=1.0, step=0.01)

    # --- CORREÇÃO: Inicializar rho_prior_bounds se não existir ---
    if 'rho_prior_bounds' not in st.session_state:
        st.session_state.rho_prior_bounds = [1400.0, 1700.0]

    if st.button("Rodar MCMC"):
        if 'experimental_data' not in st.session_state or st.session_state.experimental_data.empty:
            st.error("Por favor, insira dados experimentais no Módulo 3 primeiro.")
        else:
            with st.spinner("Rodando MCMC... Isso pode levar alguns minutos para muitas iterações."):
                # Preparar estados iniciais
                param_names = list(st.session_state.prior_bounds_gpa.keys())
                n_params_c = len(param_names)
                n_total_params = n_params_c + 1 # C_params + rho
                
                initial_states = []
                for _ in range(n_chains):
                    initial_c_params = {name: uniform.rvs(loc=st.session_state.prior_bounds_gpa[name][0], 
                        scale=st.session_state.prior_bounds_gpa[name][1] - st.session_state.prior_bounds_gpa[name][0])
                        for name in param_names}
                    initial_rho = uniform.rvs(loc=st.session_state.rho_prior_bounds[0],
                        scale=st.session_state.rho_prior_bounds[1] - st.session_state.rho_prior_bounds[0])
                    initial_state_vector = np.concatenate((get_param_vector_from_dict(initial_c_params), [initial_rho]))
                    initial_states.append(initial_state_vector)
                all_samples, all_log_posteriors, all_acceptance_rates = st.session_state.mcmc_sampler.sample(
                    initial_states, n_iter, burn_in, proposal_scale, 
                    st.session_state.experimental_data, st.session_state.prior_bounds_gpa, st.session_state.rho_prior_bounds
                )
                st.session_state.mcmc_results = {
                    'all_samples': all_samples,
                    'all_log_posteriors': all_log_posteriors,
                    'all_acceptance_rates': all_acceptance_rates,
                    'burn_in': burn_in,
                    'param_names': param_names + ['rho']
                }
                st.success("MCMC Concluído!")

        if 'mcmc_results' in st.session_state:
            results = st.session_state.mcmc_results
            all_samples = results['all_samples']
            all_log_posteriors = results['all_log_posteriors']
            all_acceptance_rates = results['all_acceptance_rates']
            burn_in = results['burn_in']
            param_names = results['param_names']

            st.subheader("Diagnósticos MCMC")
            
            # R-hat
            rhat_values = st.session_state.mcmc_sampler.calculate_rhat(all_samples, burn_in)
            if rhat_values is not None:
                st.write("##### R-hat (Gelman-Rubin)")
                rhat_df = pd.DataFrame({'Parâmetro': param_names, 'R-hat': rhat_values})
                st.dataframe(rhat_df)
                if np.any(rhat_values > 1.1):
                    st.warning("R-hat > 1.1 para alguns parâmetros, indicando possível falta de convergência. Considere mais iterações ou ajuste a escala da proposta.")
                else:
                    st.success("R-hat < 1.1 para todos os parâmetros, indicando boa convergência entre cadeias.")
            
            # ESS
            ess_values = st.session_state.mcmc_sampler.calculate_ess(all_samples, burn_in)
            if ess_values is not None:
                st.write("##### Effective Sample Size (ESS)")
                ess_df = pd.DataFrame({'Parâmetro': param_names, 'ESS': ess_values})
                st.dataframe(ess_df)
                if np.any(ess_values < 400):
                    st.warning("ESS < 400 para alguns parâmetros, indicando alta autocorrelação. Considere mais iterações ou um burn-in maior.")
                else:
                    st.success("ESS > 400 para a maioria dos parâmetros, indicando amostras efetivas suficientes.")

            st.write("##### Taxas de Aceitação")
            for i, rate in enumerate(all_acceptance_rates):
                st.write(f"Cadeia {i+1}: {rate:.2%}")
            
            st.subheader("Trace Plots e Histograma Marginal")
            
            # Plotar trace plots e histogramas para os primeiros 4 parâmetros (para não sobrecarregar)
            num_plots = min(len(param_names), 4)
            for p_idx in range(num_plots):
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Trace Plot
                for chain_idx in range(len(all_samples)):
                    axes[0].plot(all_samples[chain_idx][:, p_idx], alpha=0.7, label=f'Cadeia {chain_idx+1}')
                axes[0].axvline(burn_in, color='red', linestyle='--', label='Burn-in')
                axes[0].set_title(f"Trace Plot para {param_names[p_idx]}")
                axes[0].set_xlabel("Iteração")
                axes[0].set_ylabel(param_names[p_idx])
                axes[0].legend()
                
                # Histograma Marginal
                combined_samples = np.concatenate([s[burn_in:, p_idx] for s in all_samples])
                sns.histplot(combined_samples, kde=True, ax=axes[1], color=COLORS[p_idx % len(COLORS)])
                axes[1].set_title(f"Histograma Marginal para {param_names[p_idx]}")
                axes[1].set_xlabel(param_names[p_idx])
                axes[1].set_ylabel("Frequência")
                
                st.pyplot(fig)
            
            st.subheader("Estimativas Posteriores")
            combined_samples_all_params = np.concatenate([s[burn_in:, :] for s in all_samples], axis=0)
            posterior_means = np.mean(combined_samples_all_params, axis=0)
            posterior_stds = np.std(combined_samples_all_params, axis=0)
            
            posterior_df = pd.DataFrame({
                'Parâmetro': param_names,
                'Média Posterior': posterior_means,
                'SD Posterior': posterior_stds
            })
            st.dataframe(posterior_df)

#--- Módulo 5: Validação Completa ---
elif page == "Módulo 5: Validação Completa":
    st.title("Módulo 5: Validação Completa")
    st.write("""
    Este módulo fornece ferramentas para análise de sensibilidade, identificabilidade e validação
    do modelo Bayesiano, utilizando os resultados do MCMC.
    """)
    if 'mcmc_results' not in st.session_state:
        st.warning("Por favor, rode o MCMC no Módulo 4 primeiro para gerar amostras posteriores.")
    else:
        results = st.session_state.mcmc_results
        all_samples = results['all_samples']
        burn_in = results['burn_in']
        param_names = results['param_names']
        
        combined_samples_all_params = np.concatenate([s[burn_in:, :] for s in all_samples], axis=0)

        st.subheader("1. Análise de Sensibilidade ao Prior")
        st.write("""
        Compara a dispersão (desvio padrão) da distribuição posterior com a dispersão do prior.
        Uma razão (SD Prior / SD Posterior) significativamente maior que 1 indica que os dados
        foram informativos para aquele parâmetro.
        """)
        sensitivity_df = st.session_state.validation_tools.sensitivity_to_prior(
            combined_samples_all_params, st.session_state.prior_bounds_gpa, st.session_state.rho_prior_bounds
        )
        st.dataframe(sensitivity_df)
        st.markdown("""
        *   **Razão > 5:** Parâmetro bem-identificado (dados muito informativos).
        *   **Razão 2-5:** Parâmetro moderadamente identificado.
        *   **Razão < 2:** Parâmetro mal-identificado (dados pouco informativos, posterior próxima ao prior).
        """)

        st.subheader("2. Análise de Identificabilidade (Correlação Posterior)")
        st.write("""
        A matriz de correlação posterior revela dependências entre os parâmetros.
        Valores absolutos próximos de 1 indicam alta correlação, o que pode sugerir
        que os parâmetros não são bem-identificáveis independentemente pelos dados.
        """)
        correlation_matrix = st.session_state.validation_tools.identifiability_analysis(combined_samples_all_params)
        correlation_df = pd.DataFrame(correlation_matrix, columns=param_names, index=param_names)
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        ax_corr.set_title("Matriz de Correlação Posterior")
        st.pyplot(fig_corr)
        st.markdown("""
        *   **Correlação |ρ| > 0.9:** Alta correlação, pode indicar problemas de identificabilidade.
        *   **Correlação |ρ| < 0.5:** Baixa correlação, parâmetros mais independentes.
        """)

        st.subheader("3. Posterior Predictive Check (PPC)")
        st.write("""
        O PPC simula dados com base nas amostras posteriores do modelo e os compara com os dados experimentais observados.
        Se o modelo for adequado, os dados observados devem ser consistentes com os dados simulados.
        """)
        
        ppc_results_df, p_value_bayesian = st.session_state.validation_tools.posterior_predictive_check(
            combined_samples_all_params, st.session_state.experimental_data
        )
        st.dataframe(ppc_results_df)
        st.write(f"**P-valor Bayesiano (baseado em estatística chi-quadrado): {p_value_bayesian:.2f}**")
        st.markdown("""
        *   **P-valor Bayesiano ≈ 0.5:** O modelo prediz os dados observados de forma consistente.
        *   **P-valor Bayesiano < 0.05 ou > 0.95:** O modelo pode ser inadequado para descrever os dados.
        """)

        fig_ppc, ax_ppc = plt.subplots(figsize=(10, 6))
        ax_ppc.errorbar(ppc_results_df.index, ppc_results_df['v_exp'], yerr=ppc_results_df['sigma_exp'], 
                        fmt='o', label='Experimental', color='red', capsize=5)
        ax_ppc.errorbar(ppc_results_df.index, ppc_results_df['v_pred_mean'], yerr=ppc_results_df['v_pred_std'], 
                        fmt='x', label='Predito (Média ± SD)', color='blue', capsize=5)
        ax_ppc.set_title("Posterior Predictive Check: Velocidades")
        ax_ppc.set_xlabel("Ponto de Medição")
        ax_ppc.set_ylabel("Velocidade (m/s)")
        ax_ppc.legend()
        ax_ppc.grid(True)
        st.pyplot(fig_ppc)

        st.subheader("4. Leave-One-Out Cross-Validation (LOO-CV) - Conceitual")
        st.session_state.validation_tools.loo_cv_conceptual()

# --- Novo Módulo: Cálculo dos Módulos Elásticos ---
def calcular_modulos_elasticos(params_gpa):
    """
    Calcula os módulos elásticos principais (E1, E2, E3, G12, G13, G23, nu12, nu13, nu23)
    para um material ortotrópico a partir dos parâmetros C_ij em GPa.
    Retorna um dicionário com os valores.
    """
    # Parâmetros em GPa
    C11 = params_gpa['C11']
    C22 = params_gpa['C22']
    C33 = params_gpa['C33']
    C12 = params_gpa['C12']
    C13 = params_gpa['C13']
    C23 = params_gpa['C23']
    C44 = params_gpa['C44']
    C55 = params_gpa['C55']
    C66 = params_gpa['C66']

    # Matriz de rigidez (Voigt, 6x6)
    C = np.array([
        [C11, C12, C13, 0,   0,   0],
        [C12, C22, C23, 0,   0,   0],
        [C13, C23, C33, 0,   0,   0],
        [0,   0,   0,   C44, 0,   0],
        [0,   0,   0,   0,   C55, 0],
        [0,   0,   0,   0,   0,   C66]
    ])
    # Matriz de compliância (S = C^-1)
    S = np.linalg.inv(C)

    # Módulos de Young
    E1 = 1 / S[0, 0]
    E2 = 1 / S[1, 1]
    E3 = 1 / S[2, 2]
    # Módulos de cisalhamento
    G12 = 1 / S[5, 5]
    G13 = 1 / S[4, 4]
    G23 = 1 / S[3, 3]
    # Coeficientes de Poisson principais
    nu12 = -S[0, 1] / S[0, 0]
    nu13 = -S[0, 2] / S[0, 0]
    nu21 = -S[1, 0] / S[1, 1]
    nu23 = -S[1, 2] / S[1, 1]
    nu31 = -S[2, 0] / S[2, 2]
    nu32 = -S[2, 1] / S[2, 2]

    return {
        "E1 (GPa)": E1,
        "E2 (GPa)": E2,
        "E3 (GPa)": E3,
        "G12 (GPa)": G12,
        "G13 (GPa)": G13,
        "G23 (GPa)": G23,
        "nu12": nu12,
        "nu13": nu13,
        "nu21": nu21,
        "nu23": nu23,
        "nu31": nu31,
        "nu32": nu32
    }

# --- Módulo 6: Módulos Elásticos ---
if page == "Módulo 6: Módulos Elásticos":
    st.title("Módulo 6: Cálculo dos Módulos Elásticos")
    st.write("""
    Este módulo calcula os principais módulos elásticos (E1, E2, E3, G12, G13, G23, Poisson) a partir dos parâmetros C<sub>ij</sub> atuais.
    """, unsafe_allow_html=True)
    params_gpa = st.session_state.c_params_gpa
    modulos = calcular_modulos_elasticos(params_gpa)
    st.subheader("Resultados dos Módulos Elásticos")
    modulos_df = pd.DataFrame(modulos, index=["Valor"]).T
    st.table(modulos_df)

    # --- Escolha do formato do relatório ---
    formato = st.selectbox(
        "Formato do relatório para download",
        options=["TXT", "CSV", "PDF"],
        index=0
    )

    # --- Geração do relatório conforme formato escolhido ---
    if formato == "TXT":
        relatorio = io.StringIO()
        relatorio.write("Relatório dos Módulos Elásticos\n")
        relatorio.write("="*35 + "\n\n")
        relatorio.write("Parâmetros C_ij utilizados (GPa):\n")
        for k, v in params_gpa.items():
            relatorio.write(f"  {k}: {v:.4f}\n")
        relatorio.write("\nResultados dos módulos elásticos:\n")
        for k, v in modulos.items():
            relatorio.write(f"  {k}: {v:.4f}\n")
        relatorio_str = relatorio.getvalue()
        st.download_button(
            label="Baixar Relatório TXT",
            data=relatorio_str,
            file_name="modulos_elasticos.txt",
            mime="text/plain"
        )
    elif formato == "CSV":
        relatorio_csv = io.StringIO()
        # Parâmetros C_ij
        pd.DataFrame(params_gpa, index=["C_ij (GPa)"]).to_csv(relatorio_csv)
        relatorio_csv.write("\n")
        # Módulos elásticos
        modulos_df.to_csv(relatorio_csv)
        st.download_button(
            label="Baixar Relatório CSV",
            data=relatorio_csv.getvalue(),
            file_name="modulos_elasticos.csv",
            mime="text/csv"
        )
    elif formato == "PDF":
        # Geração do PDF usando fpdf
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "Relatório dos Módulos Elásticos", ln=True, align="C")
        pdf.ln(5)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, "Parâmetros C_ij utilizados (GPa):", ln=True)
        for k, v in params_gpa.items():
            pdf.cell(0, 8, f"  {k}: {v:.4f}", ln=True)
        pdf.ln(4)
        pdf.cell(0, 8, "Resultados dos módulos elásticos:", ln=True)
        for k, v in modulos.items():
            pdf.cell(0, 8, f"  {k}: {v:.4f}", ln=True)
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        st.download_button(
            label="Baixar Relatório PDF",
            data=pdf_bytes,
            file_name="modulos_elasticos.pdf",
            mime="application/pdf"
        )

# --- Atualizar menu lateral ---
st.sidebar.title("Navegação")
page = st.sidebar.radio("Selecione um Módulo", [
    "Módulo 1: Christoffel Solver",
    "Módulo 2: Modelo Ultrassônico",
    "Módulo 3: Likelihood Bayesiana",
    "Módulo 4: MCMC Metropolis-Hastings",
    "Módulo 5: Validação Completa",
    "Módulo 6: Módulos Elásticos"
])