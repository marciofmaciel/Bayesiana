import streamlit as st
import numpy as np
from scipy.integrate import trapezoid
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform
from scipy.optimize import minimize

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Inferência Bayesiana para Compósitos",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Inferência Bayesiana para Caracterização de Laminados Compósitos via Ultrassom")
st.markdown("Esta aplicação interativa demonstra os conceitos e etapas da inferência Bayesiana aplicada à caracterização de propriedades elásticas de compósitos usando ultrassom.")

# --- Helper Functions (Simulations) ---

# Módulo 1: Simplified Christoffel Solver (Conceptual)
def simulate_christoffel_velocities(C_vals, rho, angle_deg):
    """
    Simula velocidades de onda para um material ortotrópico simplificado.
    Não é um solver Christoffel completo, mas ilustra a dependência angular.
    Assume propagação no plano 1-2.
    """
    angle_rad = np.deg2rad(angle_deg)
    
    # Simplified C_ij mapping for orthotropic material
    C11, C22, C12, C66 = C_vals
    
    # Simplified velocity calculation (conceptual, not exact Christoffel)
    # Longitudinal-like velocity
    v_longitudinal = np.sqrt((C11 * np.cos(angle_rad)**2 + C22 * np.sin(angle_rad)**2 + 2 * C12 * np.cos(angle_rad) * np.sin(angle_rad) + C66) / rho)
    # Shear-like velocity (simplified)
    v_shear = np.sqrt((C66 * np.cos(angle_rad)**2 + C66 * np.sin(angle_rad)**2) / rho) # Simplified, often C44, C55, C66 are different
    
    return v_longitudinal, v_shear

# Módulo 2: Ultrasonic Signal Simulation
def simulate_ultrasonic_signal(frequency_MHz, duration_us, noise_level, TOF_us):
    """Simula um sinal ultrassônico com ruído e um pulso."""
    sampling_rate_MHz = 100 # 100 MS/s
    time = np.linspace(0, duration_us, int(duration_us * sampling_rate_MHz))
    
    # Simulate a noisy baseline
    signal = np.random.normal(0, noise_level, len(time))
    
    # Simulate a pulse at TOF
    pulse_start_idx = int(TOF_us * sampling_rate_MHz)
    if pulse_start_idx < len(time):
        pulse_duration_samples = int(1.5 * sampling_rate_MHz / frequency_MHz) # ~1.5 cycles
        pulse_amplitude = 1.0
        
        # Ricker wavelet or simple sine burst
        t_pulse = np.linspace(-pulse_duration_samples / sampling_rate_MHz / 2, 
                              pulse_duration_samples / sampling_rate_MHz / 2, 
                              pulse_duration_samples)
        
        # Simple sine burst with Gaussian envelope
        envelope = np.exp(-t_pulse**2 / (2 * (0.2 / frequency_MHz)**2))
        pulse = pulse_amplitude * np.sin(2 * np.pi * frequency_MHz * t_pulse) * envelope
        
        end_idx = pulse_start_idx + len(pulse)
        if end_idx > len(time):
            pulse = pulse[:len(time) - pulse_start_idx]
            end_idx = len(time)
            
        signal[pulse_start_idx:end_idx] += pulse
        
    return time, signal

def calculate_velocity_and_uncertainty(h_mm, delta_h_mm, TOF_us, delta_TOF_us, technique="Transmissão"):
    """Calcula velocidade e propaga incertezas."""
    h_m = h_mm / 1000
    delta_h_m = delta_h_mm / 1000
    TOF_s = TOF_us / 1e6
    delta_TOF_s = delta_TOF_us / 1e6

    if technique == "Transmissão":
        v = h_m / TOF_s
        # Error propagation for v = h/TOF
        delta_v_rel = np.sqrt((delta_h_m / h_m)**2 + (delta_TOF_s / TOF_s)**2)
    else: # Reflexão
        v = (2 * h_m) / TOF_s
        # Error propagation for v = 2h/TOF
        delta_v_rel = np.sqrt((delta_h_m / h_m)**2 + (delta_TOF_s / TOF_s)**2)
        
    delta_v = v * delta_v_rel
    return v, delta_v

# Módulo 3: Bayesian Inference Concepts
def simulate_likelihood_prior_posterior(v_exp, sigma_exp, prior_mean, prior_std, param_range=(0, 200)):
    """Simula distribuições de prior, likelihood e posterior para um único parâmetro."""
    param_values = np.linspace(param_range[0], param_range[1], 500)
    
    # Prior (Gaussian for simplicity)
    prior_dist = norm.pdf(param_values, loc=prior_mean, scale=prior_std)
    
    # Likelihood (assuming v_pred = param_value for simplicity)
    # In a real scenario, v_pred would come from the forward model
    likelihood_dist = norm.pdf(v_exp, loc=param_values, scale=sigma_exp)
    
    # Posterior (unnormalized)
    posterior_unnorm = likelihood_dist * prior_dist
    
    # Normalize posterior for plotting
    posterior_dist = posterior_unnorm / trapezoid(posterior_unnorm, param_values)
    
    return param_values, prior_dist, likelihood_dist, posterior_dist

# Módulo 4: MCMC Simulation
def simulate_mcmc_chain(num_iterations, step_size, true_value, initial_value, likelihood_std, prior_mean, prior_std):
    """Simula uma cadeia MCMC Metropolis-Hastings para um único parâmetro."""
    samples = np.zeros(num_iterations)
    current_param = initial_value
    accepted_count = 0

    # Simplified log-posterior function for a single parameter
    def log_posterior_func(param):
        if not (0 < param < 200): # Simple bounds
            return -np.inf
        log_prior = norm.logpdf(param, loc=prior_mean, scale=prior_std)
        # Simulate likelihood: assume true_value is the "measured" value
        log_likelihood = norm.logpdf(true_value, loc=param, scale=likelihood_std)
        return log_prior + log_likelihood

    current_log_post = log_posterior_func(current_param)

    for i in range(num_iterations):
        # Propose a new parameter value
        proposed_param = current_param + np.random.normal(0, step_size)
        
        # Calculate log-posterior for proposed value
        proposed_log_post = log_posterior_func(proposed_param)
        
        # Calculate acceptance ratio
        alpha = np.exp(proposed_log_post - current_log_post)
        
        # Accept or reject
        if np.random.rand() < alpha:
            current_param = proposed_param
            current_log_post = proposed_log_post
            accepted_count += 1
        
        samples[i] = current_param
        
    acceptance_rate = accepted_count / num_iterations
    
    # Simulate R_hat and ESS (conceptual values for demonstration)
    r_hat = 1.0 + (np.random.rand() * 0.2 if acceptance_rate < 0.2 or acceptance_rate > 0.5 else np.random.rand() * 0.05)
    ess = num_iterations * (acceptance_rate * 0.5) # Simplified relation
    
    return samples, acceptance_rate, r_hat, ess

# Módulo 5: Sensitivity and Validation
def simulate_posterior_samples(num_samples, prior_mean, prior_std, true_value, likelihood_std, correlation_strength=0.0):
    """Simula amostras posteriores para 2 parâmetros com correlação."""
    # Simulate a more informative posterior than prior
    posterior_std = prior_std / (1 + np.random.rand() * 2) # Posterior is narrower
    
    # Simulate samples for C11
    C11_samples = np.random.normal(true_value, posterior_std, num_samples)
    
    # Simulate samples for C12, potentially correlated with C11
    if correlation_strength != 0:
        # Create correlated samples
        mean = [true_value, true_value * 0.05] # C12 is typically much smaller than C11
        cov = [[posterior_std**2, correlation_strength * posterior_std * (posterior_std/5)], 
               [correlation_strength * posterior_std * (posterior_std/5), (posterior_std/5)**2]]
        
        samples_2d = np.random.multivariate_normal(mean, cov, num_samples)
        C11_samples = samples_2d[:, 0]
        C12_samples = samples_2d[:, 1]
    else:
        C12_samples = np.random.normal(true_value * 0.05, posterior_std / 5, num_samples)

    # Simulate prior samples for comparison
    C11_prior_samples = np.random.normal(prior_mean, prior_std, num_samples)
    C12_prior_samples = np.random.normal(prior_mean * 0.05, prior_std / 5, num_samples)
    
    return C11_samples, C12_samples, C11_prior_samples, C12_prior_samples

# --- Module Functions ---

def module1_fundamentals():
    st.header("Módulo 1: Fundamentos de Propagação de Ondas")
    st.markdown("""
    Este módulo explora como as propriedades elásticas de um compósito anisotrópico afetam a velocidade de propagação das ondas ultrassônicas.
    A Equação de Christoffel é a base para relacionar as constantes elásticas (C_ij) com as velocidades de onda em diferentes direções.
    """)

    st.subheader("Parâmetros do Material (Simplificado)")
    col1, col2 = st.columns(2)
    with col1:
        C11 = st.slider("C₁₁ (GPa)", 50, 200, 140, 5) * 1e9
        C22 = st.slider("C₂₂ (GPa)", 5, 20, 10, 1) * 1e9
        C12 = st.slider("C₁₂ (GPa)", 2, 10, 5, 1) * 1e9
    with col2:
        C66 = st.slider("C₆₆ (GPa)", 3, 15, 7, 1) * 1e9
        rho = st.slider("Densidade (kg/m³)", 1000, 2000, 1550, 50)
        
    st.subheader("Direção de Propagação")
    angle_deg = st.slider("Ângulo de Propagação (graus no plano 1-2)", 0, 90, 0, 5)

    if st.button("Calcular Velocidades"):
        C_vals = (C11, C22, C12, C66)
        v_long, v_shear = simulate_christoffel_velocities(C_vals, rho, angle_deg)
        
        st.write(f"**Velocidade Longitudinal (simulada):** {v_long/1000:.2f} km/s")
        st.write(f"**Velocidade Cisalhante (simulada):** {v_shear/1000:.2f} km/s")
        
        st.markdown("---")
        st.subheader("Dependência Angular da Velocidade (Exemplo Conceitual)")
        angles = np.linspace(0, 90, 19)
        v_long_plot = []
        v_shear_plot = []
        for a in angles:
            vl, vs = simulate_christoffel_velocities(C_vals, rho, a)
            v_long_plot.append(vl / 1000)
            v_shear_plot.append(vs / 1000)
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(angles, v_long_plot, label="Velocidade Longitudinal", marker='o')
        ax.plot(angles, v_shear_plot, label="Velocidade Cisalhante", marker='x')
        ax.set_xlabel("Ângulo de Propagação (graus)")
        ax.set_ylabel("Velocidade (km/s)")
        ax.set_title("Velocidade de Onda vs. Ângulo de Propagação (Simulado)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

def module2_ultrasound_measurement():
    st.header("Módulo 2: Medição por Ultrassom e Incertezas")
    st.markdown("""
    Este módulo demonstra como as velocidades de onda são extraídas de sinais ultrassônicos e como as incertezas experimentais são propagadas.
    """)

    st.subheader("Parâmetros da Medição")
    col1, col2 = st.columns(2)
    with col1:
        h_mm = st.slider("Espessura da Amostra (mm)", 1.0, 10.0, 5.0, 0.1)
        delta_h_mm = st.slider("Incerteza na Espessura (mm)", 0.01, 0.1, 0.05, 0.01)
    with col2:
        TOF_us = st.slider("Tempo de Voo (TOF) (µs)", 1.0, 5.0, 2.5, 0.1)
        delta_TOF_us = st.slider("Incerteza no TOF (µs)", 0.001, 0.1, 0.02, 0.001)
    
    technique = st.radio("Técnica de Medição", ["Transmissão", "Reflexão"])

    if st.button("Calcular Velocidade e Incerteza"):
        v, delta_v = calculate_velocity_and_uncertainty(h_mm, delta_h_mm, TOF_us, delta_TOF_us, technique)
        
        st.write(f"**Velocidade Calculada:** {v:.2f} m/s")
        st.write(f"**Incerteza na Velocidade:** ± {delta_v:.2f} m/s ({delta_v/v*100:.2f}%)")
        
        st.markdown("---")
        st.subheader("Sinal Ultrassônico Simulado")
        
        freq_MHz = st.slider("Frequência do Transdutor (MHz)", 1.0, 10.0, 5.0, 0.5)
        noise_lvl = st.slider("Nível de Ruído", 0.01, 0.5, 0.1, 0.01)
        
        time_signal, signal_data = simulate_ultrasonic_signal(freq_MHz, TOF_us * 2, noise_lvl, TOF_us)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_signal, signal_data)
        ax.axvline(x=TOF_us, color='r', linestyle='--', label=f'TOF = {TOF_us} µs')
        ax.set_xlabel("Tempo (µs)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Sinal Ultrassônico Simulado")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

def module3_bayesian_inference():
    st.header("Módulo 3: Inferência Bayesiana: Conceitos")
    st.markdown("""
    Este módulo ilustra os componentes fundamentais da inferência Bayesiana: o Prior, a Likelihood e a Posterior.
    Vamos simular a estimativa de um único parâmetro (e.g., C₁₁) a partir de uma medição.
    """)

    st.subheader("Dados Observados (Simulados)")
    v_exp = st.slider("Velocidade Medida (v_exp, m/s)", 1000, 10000, 7400, 100)
    sigma_exp = st.slider("Incerteza da Medição (σ_exp, m/s)", 10, 200, 50, 10)

    st.subheader("Conhecimento Pré-existente (Prior)")
    prior_mean = st.slider("Média do Prior (m/s)", 1000, 10000, 7000, 100)
    prior_std = st.slider("Desvio Padrão do Prior (m/s)", 100, 2000, 1000, 100)
    
    param_range_min = min(v_exp - 3*sigma_exp, prior_mean - 3*prior_std) - 500
    param_range_max = max(v_exp + 3*sigma_exp, prior_mean + 3*prior_std) + 500
    
    if st.button("Visualizar Distribuições"):
        param_values, prior_dist, likelihood_dist, posterior_dist = \
            simulate_likelihood_prior_posterior(v_exp, sigma_exp, prior_mean, prior_std, 
                                                param_range=(param_range_min, param_range_max))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(param_values, prior_dist, label="Prior", linestyle='--')
        ax.plot(param_values, likelihood_dist, label="Likelihood")
        ax.plot(param_values, posterior_dist, label="Posterior", linewidth=2, color='red')
        
        ax.axvline(x=v_exp, color='gray', linestyle=':', label=f'v_exp = {v_exp} m/s')
        
        ax.set_xlabel("Valor do Parâmetro (e.g., C₁₁ equivalente em m/s)")
        ax.set_ylabel("Densidade de Probabilidade")
        ax.set_title("Prior, Likelihood e Posterior (Conceitual)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown("---")
        st.subheader("Interpretação")
        st.write(f"**Média do Prior:** {prior_mean:.0f} m/s, **SD do Prior:** {prior_std:.0f} m/s")
        st.write(f"**Média da Likelihood (dada v_exp):** {v_exp:.0f} m/s, **SD da Likelihood:** {sigma_exp:.0f} m/s")
        
        posterior_mean = trapezoid(param_values * posterior_dist, param_values)
        posterior_std = np.sqrt(trapezoid((param_values - posterior_mean)**2 * posterior_dist, param_values))
        
        st.write(f"**Média da Posterior:** {posterior_mean:.0f} m/s, **SD da Posterior:** {posterior_std:.0f} m/s")
        st.markdown(f"""
        - O **Prior** representa nosso conhecimento inicial sobre o parâmetro.
        - A **Likelihood** mostra quão prováveis são os dados observados para cada valor possível do parâmetro.
        - A **Posterior** é a combinação do Prior e da Likelihood, representando nosso conhecimento atualizado sobre o parâmetro após observar os dados.
        - Observe como a posterior é mais estreita que o prior, indicando uma **redução da incerteza** devido aos dados.
        """)

def module4_mcmc_algorithms():
    st.header("Módulo 4: Algoritmos MCMC em Ação")
    st.markdown("""
    Este módulo demonstra o funcionamento do algoritmo Metropolis-Hastings para amostrar a distribuição posterior.
    """)

    st.subheader("Configuração MCMC (para um único parâmetro)")
    col1, col2 = st.columns(2)
    with col1:
        num_iterations = st.slider("Número de Iterações", 1000, 50000, 10000, 1000)
        step_size = st.slider("Tamanho do Passo (Step Size)", 0.1, 100.0, 10.0, 0.1)
        initial_value = st.slider("Valor Inicial da Cadeia", 1000, 10000, 6000, 100)
    with col2:
        true_value = st.slider("Valor 'Verdadeiro' Simulado (para Likelihood)", 1000, 10000, 7400, 100)
        likelihood_std = st.slider("Desvio Padrão da Likelihood", 10, 200, 50, 10)
        prior_mean = st.slider("Média do Prior (para MCMC)", 1000, 10000, 7000, 100)
        prior_std = st.slider("Desvio Padrão do Prior (para MCMC)", 100, 2000, 1000, 100)

    if st.button("Rodar MCMC"):
        samples, acceptance_rate, r_hat, ess = simulate_mcmc_chain(
            num_iterations, step_size, true_value, initial_value, likelihood_std, prior_mean, prior_std
        )
        
        st.markdown("---")
        st.subheader("Resultados da Cadeia MCMC")
        
        col_metrics1, col_metrics2 = st.columns(2)
        with col_metrics1:
            st.metric("Taxa de Aceitação", f"{acceptance_rate*100:.2f}%")
            st.metric("R̂ (Gelman-Rubin)", f"{r_hat:.2f}")
        with col_metrics2:
            st.metric("ESS (Effective Sample Size)", f"{int(ess)}")
            
        st.markdown("""
        **Interpretação das Métricas:**
        - **Taxa de Aceitação:** Idealmente entre 20-40%. Baixa demais indica passos muito grandes; alta demais indica passos muito pequenos.
        - **R̂ (Gelman-Rubin):** Deve ser próximo de 1.00 (tipicamente < 1.05) para indicar convergência.
        - **ESS:** Número de amostras independentes efetivas. Deve ser alto o suficiente (ex: > 400) para inferências confiáveis.
        """)

        # Trace Plot
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(samples)
        ax1.set_xlabel("Iteração")
        ax1.set_ylabel("Valor do Parâmetro")
        ax1.set_title("Trace Plot da Cadeia MCMC")
        ax1.grid(True)
        st.pyplot(fig1)
        plt.close(fig1)

        # Histograma da Posterior
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.histplot(samples[int(num_iterations*0.2):], kde=True, ax=ax2, color='skyblue') # Discard burn-in
        ax2.axvline(x=true_value, color='red', linestyle='--', label="Valor 'Verdadeiro' Simulado")
        ax2.set_xlabel("Valor do Parâmetro")
        ax2.set_ylabel("Frequência")
        ax2.set_title("Distribuição Posterior Amostrada (após Burn-in)")
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)

def module5_sensitivity_validation():
    st.header("Módulo 5: Análise de Sensibilidade e Validação")
    st.markdown("""
    Este módulo explora como avaliar a identificabilidade dos parâmetros, o impacto das correlações e a validação do modelo.
    """)

    st.subheader("Simulação de Amostras Posteriores")
    num_samples = st.slider("Número de Amostras Posteriores", 1000, 10000, 5000, 1000)
    prior_mean_C11 = st.slider("Média do Prior C₁₁", 100, 200, 140, 5)
    prior_std_C11 = st.slider("Desvio Padrão do Prior C₁₁", 10, 50, 30, 5)
    true_value_C11 = st.slider("Valor 'Verdadeiro' Simulado C₁₁", 100, 200, 138, 5)
    likelihood_std_C11 = st.slider("Desvio Padrão da Likelihood C₁₁", 1, 10, 3, 1)
    
    correlation_strength = st.slider("Força da Correlação C₁₁-C₁₂", -0.99, 0.99, -0.7, 0.05)

    if st.button("Analisar Sensibilidade e Correlação"):
        C11_post_samples, C12_post_samples, C11_prior_samples, C12_prior_samples = \
            simulate_posterior_samples(num_samples, prior_mean_C11, prior_std_C11, 
                                       true_value_C11, likelihood_std_C11, correlation_strength)
        
        st.markdown("---")
        st.subheader("1. Identificabilidade (Comparação Prior vs. Posterior)")
        
        col_id1, col_id2 = st.columns(2)
        with col_id1:
            st.write(f"**C₁₁:**")
            st.write(f"SD Prior: {np.std(C11_prior_samples):.2f}")
            st.write(f"SD Posterior: {np.std(C11_post_samples):.2f}")
            sd_ratio_C11 = np.std(C11_prior_samples) / np.std(C11_post_samples)
            st.write(f"Razão SD (Prior/Posterior): {sd_ratio_C11:.2f}")
            st.markdown(f"**Interpretação C₁₁:** {'Bem identificável' if sd_ratio_C11 > 5 else ('Moderadamente identificável' if sd_ratio_C11 > 2 else 'Mal identificável')}")
        
        with col_id2:
            st.write(f"**C₁₂:**")
            st.write(f"SD Prior: {np.std(C12_prior_samples):.2f}")
            st.write(f"SD Posterior: {np.std(C12_post_samples):.2f}")
            sd_ratio_C12 = np.std(C12_prior_samples) / np.std(C12_post_samples)
            st.write(f"Razão SD (Prior/Posterior): {sd_ratio_C12:.2f}")
            st.markdown(f"**Interpretação C₁₂:** {'Bem identificável' if sd_ratio_C12 > 5 else ('Moderadamente identificável' if sd_ratio_C12 > 2 else 'Mal identificável')}")

        fig_id, ax_id = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(C11_prior_samples, kde=True, color='blue', label='Prior C₁₁', ax=ax_id[0], stat='density', alpha=0.5)
        sns.histplot(C11_post_samples, kde=True, color='red', label='Posterior C₁₁', ax=ax_id[0], stat='density', alpha=0.7)
        ax_id[0].set_title("Prior vs. Posterior para C₁₁")
        ax_id[0].legend()

        sns.histplot(C12_prior_samples, kde=True, color='blue', label='Prior C₁₂', ax=ax_id[1], stat='density', alpha=0.5)
        sns.histplot(C12_post_samples, kde=True, color='red', label='Posterior C₁₂', ax=ax_id[1], stat='density', alpha=0.7)
        ax_id[1].set_title("Prior vs. Posterior para C₁₂")
        ax_id[1].legend()
        st.pyplot(fig_id)
        plt.close(fig_id)

        st.markdown("---")
        st.subheader("2. Impacto da Correlação Extrema")
        
        # Calculate correlation
        correlation_matrix = np.corrcoef(C11_post_samples, C12_post_samples)
        st.write(f"**Correlação Posterior (C₁₁, C₁₂):** {correlation_matrix[0, 1]:.2f}")
        
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=C11_post_samples, y=C12_post_samples, ax=ax_corr, alpha=0.3)
        ax_corr.set_xlabel("C₁₁")
        ax_corr.set_ylabel("C₁₂")
        ax_corr.set_title("Scatter Plot das Amostras Posteriores (C₁₁ vs C₁₂)")
        st.pyplot(fig_corr)
        plt.close(fig_corr)
        
        st.markdown(f"""
        Uma correlação de **{correlation_matrix[0, 1]:.2f}** entre C₁₁ e C₁₂ indica uma forte dependência.
        - **Impacto nas Marginais:** As distribuições marginais (histogramas individuais) podem parecer razoáveis, mas não capturam o "trade-off" entre os parâmetros.
        - **Impacto nos Intervalos Conjuntos:** A região de credibilidade conjunta (visível no scatter plot) é alongada e estreita. Isso significa que, embora individualmente C₁₁ e C₁₂ possam ter uma certa faixa de valores, apenas combinações específicas ao longo da linha de correlação são plausíveis. Ignorar essa correlação pode levar a conclusões enganosas sobre a variabilidade real dos parâmetros.
        """)

        st.markdown("---")
        st.subheader("3. Posterior Predictive Check (PPC)")
        st.markdown("""
        O PPC verifica se o modelo é capaz de gerar dados semelhantes aos observados.
        Aqui, simulamos dados preditivos e os comparamos com um valor "observado" simulado.
        """)
        
        # Simulate observed data for PPC
        sim_observed_v = np.random.normal(true_value_C11, likelihood_std_C11)
        
        # Simulate predictive data from posterior samples
        sim_predictive_v = np.random.normal(C11_post_samples, likelihood_std_C11)
        
        fig_ppc, ax_ppc = plt.subplots(figsize=(10, 6))
        sns.histplot(sim_predictive_v, kde=True, color='green', label='Dados Preditivos', ax=ax_ppc, stat='density', alpha=0.7)
        ax_ppc.axvline(x=sim_observed_v, color='red', linestyle='--', label='Dado Observado Simulado')
        ax_ppc.set_xlabel("Velocidade (m/s)")
        ax_ppc.set_ylabel("Densidade")
        ax_ppc.set_title("Posterior Predictive Check (PPC) para C₁₁")
        ax_ppc.legend()
        st.pyplot(fig_ppc)
        plt.close(fig_ppc)
        
        # Calculate p-value for PPC (conceptual)
        p_value_ppc = np.mean(sim_predictive_v > sim_observed_v)
        st.markdown(f"""
        - O **Dado Observado Simulado** é o valor que o modelo tenta explicar.
        - Os **Dados Preditivos** são gerados usando os parâmetros amostrados da posterior.
        - Se o dado observado cair dentro da distribuição dos dados preditivos (especialmente perto do centro), o modelo é considerado **adequado**.
        - Um p-valor preditivo de **{p_value_ppc:.2f}** (proporção de dados preditivos maiores que o observado) indica que o modelo é {'adequado' if 0.05 < p_value_ppc < 0.95 else 'potencialmente inadequado'}.
        """)


# --- Main App Navigation ---
st.sidebar.title("Navegação")
selected_module = st.sidebar.radio(
    "Escolha um Módulo",
    [
        "Módulo 1: Fundamentos",
        "Módulo 2: Medição Ultrassônica",
        "Módulo 3: Inferência Bayesiana",
        "Módulo 4: Algoritmos MCMC",
        "Módulo 5: Sensibilidade e Validação"
    ]
)

if selected_module == "Módulo 1: Fundamentos":
    module1_fundamentals()
elif selected_module == "Módulo 2: Medição Ultrassônica":
    module2_ultrasound_measurement()
elif selected_module == "Módulo 3: Inferência Bayesiana":
    module3_bayesian_inference()
elif selected_module == "Módulo 4: Algoritmos MCMC":
    module4_mcmc_algorithms()
elif selected_module == "Módulo 5: Sensibilidade e Validação":
    module5_sensitivity_validation()