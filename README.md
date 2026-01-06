Aqui está o arquivo README.md completo e detalhado para a aplicação Streamlit, conforme solicitado.

`markdown
🔬 Inferência Bayesiana para Caracterização de Propriedades Elásticas de Compósitos via Ultrassom

Visão Geral do Projeto

Este projeto implementa uma aplicação interativa em Streamlit para a caracterização de propriedades elásticas de laminados compósitos anisotrópicos utilizando inferência Bayesiana e dados de ultrassom. Diferente de abordagens simplificadas, esta aplicação foca em implementações fisicamente precisas e exatas dos modelos subjacentes, desde a propagação de ondas até a inferência estatística.

O objetivo é fornecer uma ferramenta robusta para pesquisadores, engenheiros e estudantes na área de ciência dos materiais, ensaios não destrutivos (END) e mecânica computacional, permitindo a exploração interativa dos conceitos e a aplicação prática da inferência Bayesiana para quantificar propriedades elásticas e suas incertezas.

Funcionalidades Principais:
*   Módulo 1: Solver exato da Equação de Christoffel para materiais ortotrópicos.
*   Módulo 2: Simulação realista de medições ultrassônicas, incluindo atenuação e dispersão.
*   Módulo 3: Formulação da Likelihood Bayesiana com tratamento rigoroso de incertezas experimentais.
*   Módulo 4: Algoritmo MCMC Metropolis-Hastings robusto com diagnósticos de convergência.
*   Módulo 5: Análise de sensibilidade, identificabilidade de parâmetros e validação do modelo.

Requisitos do Sistema

*   Sistema Operacional: Windows, macOS ou Linux.
*   Python: Versão 3.8 ou superior.
*   Memória RAM: Mínimo de 8GB (16GB ou mais recomendado para execuções MCMC longas).
*   Espaço em Disco: Aproximadamente 500MB para o ambiente e bibliotecas.

Instruções de Instalação Passo a Passo

Siga os passos abaixo para configurar e executar a aplicação:

1.  Clone o Repositório (ou baixe os arquivos):
    `bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
    `
    (Se você baixou os arquivos diretamente, navegue até a pasta onde eles estão.)

2.  Crie um Ambiente Virtual (Recomendado):
    É uma boa prática isolar as dependências do projeto.
    `bash
    python -m venv venv
    `

3.  Ative o Ambiente Virtual:
    *   No Windows:
        `bash
        .\venv\Scripts\activate
        `
    *   No macOS/Linux:
        `bash
        source venv/bin/activate
        `

4.  Instale as Dependências:
    Com o ambiente virtual ativado, instale todas as bibliotecas necessárias usando o requirements.txt fornecido:
    `bash
    pip install -r requirements.txt
    `
    (Se o requirements.txt não foi fornecido explicitamente, você pode criá-lo com as seguintes dependências e depois executar o comando acima):
    `
    # requirements.txt
    streamlit>=1.30.0
    numpy>=1.26.0
    scipy>=1.11.0
    matplotlib>=3.8.0
    seaborn>=0.13.0
    pandas>=2.1.0
    `

Como Executar a Aplicação

Após a instalação das dependências, execute a aplicação Streamlit a partir do diretório raiz do projeto:

`bash
streamlit run main_app.py
`

Isso abrirá automaticamente a aplicação no seu navegador padrão, geralmente em http://localhost:8501.

Descrição Detalhada de Cada Módulo

A aplicação é estruturada em 5 módulos principais, acessíveis através da barra lateral (sidebar) do Streamlit. Cada módulo implementa uma parte crucial do processo de caracterização, com foco na precisão física e matemática.

Módulo 1: Fundamentos e Modelo Direto (Solver Christoffel Exato)

Este módulo estabelece a base teórica para a propagação de ondas em materiais anisotrópicos.

*   Propósito: Calcular as velocidades de fase teóricas (qP, qS1, qS2) e os vetores de polarização para qualquer direção de propagação em um material ortotrópico, dadas suas constantes elásticas e densidade.
*   Implementação Exata:
    *   Tensor de Elasticidade (C_ijkl): Utiliza a matriz de rigidez elástica completa para um material ortotrópico (9 constantes independentes: C₁₁, C₁₂, C₁₃, C₂₂, C₂₃, C₃₃, C₄₄, C₅₅, C₆₆) na notação de Voigt.
    *   Equação de Christoffel: Para uma direção de propagação n = [n₁, n₂, n₃], o tensor acústico Γ_ik = C_ijkl n_j n_l é construído.
    *   Problema de Autovalores: A equação (Γ_ik - ρv²δ_ik)A_k = 0 é resolvida como um problema de autovalores para a matriz 3x3 Γ. Os autovalores ρv² fornecem as três velocidades de fase v, e os autovetores A correspondem aos vetores de polarização.
*   Interação no Streamlit:
    *   Sliders para ajustar as 9 constantes elásticas (C_ij) e a densidade (ρ).
    *   Sliders para definir a direção de propagação (ângulos θ e φ).
    *   Gráficos polares ou cartesianos mostrando as velocidades de fase em função do ângulo.
    *   Exibição dos vetores de polarização para cada modo de onda.

Módulo 2: Modelo Ultrassônico Realista e Medição

Este módulo simula o processo de medição ultrassônica, gerando dados realistas e extraindo informações cruciais.

*   Propósito: Simular um sinal ultrassônico (A-scan) que atravessa uma amostra compósita, considerando efeitos físicos reais, e extrair o Tempo de Voo (TOF) com suas incertezas.
*   Implementação Exata:
    *   Propagação de Ondas: O modelo considera a propagação de um pulso de banda larga através da espessura da amostra, utilizando as velocidades calculadas no Módulo 1.
    *   Atenuação Viscoelástica: O sinal é atenuado exponencialmente com a distância percorrida, com coeficientes de atenuação que podem ser dependentes da frequência e do material.
    *   Dispersão: A velocidade de fase pode variar com a frequência, resultando em distorção do pulso. O modelo pode incorporar um termo dispersivo.
    *   Acoplamento Transdutor-Amostra: Efeitos de interface e ruído são adicionados para simular condições experimentais.
    *   Extração de TOF: Utiliza o método de correlação cruzada entre o pulso de referência (emitido) e o pulso recebido para determinar o TOF de forma robusta, minimizando o impacto de ruído e distorção.
    *   Quantificação de Incertezas: As incertezas na espessura (δh), no TOF (δTOF) e na temperatura (δT) são combinadas para estimar a incerteza total na velocidade medida (δv).
*   Interação no Streamlit:
    *   Sliders para ajustar parâmetros de simulação (frequência central, largura de banda, atenuação, nível de ruído).
    *   Input para espessura da amostra e suas incertezas.
    *   Gráfico do A-scan simulado (sinal no tempo).
    *   Exibição do TOF extraído e da velocidade experimental calculada com sua incerteza.

Módulo 3: Inferência Bayesiana - Likelihood e Priors

Este módulo formula o problema inverso Bayesiano, conectando as medições com os parâmetros a serem inferidos.

*   Propósito: Definir a função de verossimilhança (likelihood) que quantifica a probabilidade de observar os dados experimentais dadas as constantes elásticas, e especificar as distribuições a priori (priors) para essas constantes.
*   Implementação Precisa:
    *   Likelihood Gaussiana: Assume que os erros de medição seguem uma distribuição Gaussiana. Para N medições independentes, a likelihood é o produto das probabilidades individuais:
        P(v_med | C) = Πᵢ (1 / √(2πσᵢ²)) * exp(-[v_med,ᵢ - v_pred,ᵢ(C)]² / (2σᵢ²))
        Onde v_pred,ᵢ(C) é a velocidade prevista pelo Modelo Direto (Módulo 1) para as constantes C, e σᵢ é a incerteza total da i-ésima medição (calculada no Módulo 2).
    *   Incorporação de Incertezas: As incertezas em h, TOF, T e ρ são propagadas para σᵢ, garantindo que a likelihood reflita a precisão real dos dados.
    *   Priors: Permite a definição de priors uniformes ou Gaussianos para cada constante elástica, refletindo o conhecimento prévio ou restrições físicas.
*   Interação no Streamlit:
    *   Input para as velocidades experimentais e suas incertezas (pode ser preenchido com dados simulados do Módulo 2).
    *   Controles para definir os limites (uniforme) ou média/desvio padrão (Gaussiano) para cada prior de C_ij.
    *   Visualização das distribuições prior.
    *   Cálculo e exibição do valor da log-likelihood para um conjunto de constantes elásticas.

Módulo 4: MCMC (Metropolis-Hastings Robusto)

Este módulo executa o coração da inferência Bayesiana, amostrando a distribuição posterior.

*   Propósito: Utilizar o algoritmo Markov Chain Monte Carlo (MCMC) Metropolis-Hastings para gerar amostras da distribuição posterior P(C | v_med), que representa a probabilidade das constantes elásticas dadas as medições.
*   Implementação Robusta:
    *   Metropolis-Hastings com Adaptação: O algoritmo gera propostas de novos estados C_proposto a partir de uma distribuição de proposta (e.g., Gaussiana). A taxa de aceitação α = min(1, P(v_med|C_proposto)P(C_proposto) / P(v_med|C_atual)P(C_atual)) determina se o novo estado é aceito. A distribuição de proposta é adaptada durante a fase de "burn-in" para otimizar a taxa de aceitação (tipicamente entre 20-40%).
    *   Múltiplas Cadeias Paralelas: Executa várias cadeias MCMC independentes a partir de diferentes pontos de partida para garantir a exploração completa do espaço de parâmetros e facilitar os diagnósticos de convergência.
    *   Diagnósticos de Convergência:
        *   Trace Plots: Gráficos da evolução de cada parâmetro ao longo das iterações.
        *   Estatística de Gelman-Rubin (R̂): Compara a variância entre e dentro das cadeias. R̂ < 1.1 indica boa convergência.
        *   Effective Sample Size (ESS): Estima o número de amostras independentes equivalentes, considerando a autocorrelação. ESS > 400 por parâmetro é desejável.
        *   Autocorrelação: Mede a dependência entre amostras consecutivas.
    *   Burn-in e Thinning: As primeiras iterações (burn-in) são descartadas para remover a dependência da inicialização. O "thinning" (amostragem a cada N iterações) reduz a autocorrelação e o tamanho do arquivo de amostras.
*   Interação no Streamlit:
    *   Controles para número de iterações, burn-in, thinning e número de cadeias.
    *   Botão para iniciar a execução do MCMC.
    *   Exibição em tempo real (ou após conclusão) de trace plots, histogramas marginais das posteriores, gráficos de correlação entre parâmetros.
    *   Tabela de diagnósticos (R̂, ESS, taxa de aceitação).

Módulo 5: Análise de Sensibilidade e Validação

Este módulo finaliza o processo, avaliando a qualidade e a confiabilidade dos resultados da inferência.

*   Propósito: Avaliar a robustez da inferência, a identificabilidade dos parâmetros e a adequação do modelo aos dados, além de comparar os resultados com métodos independentes.
*   Implementação Completa:
    *   Análise de Sensibilidade ao Prior: Compara as distribuições posteriores obtidas com diferentes priors (e.g., informativo vs. não-informativo) para verificar se os dados dominam a inferência.
    *   Identificabilidade de Parâmetros: Avalia o quanto cada parâmetro é bem determinado pelos dados, comparando o desvio padrão da posterior com o desvio padrão do prior (razão SD posterior/prior). Uma razão baixa indica boa identificabilidade.
    *   Posterior Predictive Check (PPC): Simula novos dados usando as amostras da posterior e compara a distribuição desses dados simulados com os dados observados. Ajuda a verificar se o modelo captura as características dos dados.
    *   Comparação com Métodos Destrutivos: Permite inserir resultados de ensaios mecânicos destrutivos (e.g., tração para C₁₁) e compara-os com as estimativas Bayesianas, calculando um z-score para avaliar a consistência.
    *   Leave-One-Out Cross-Validation (LOO-CV): (Conceitual ou simplificado) Avalia a capacidade preditiva do modelo para cada ponto de dado, ajudando a identificar outliers ou áreas onde o modelo falha.
*   Interação no Streamlit:
    *   Tabelas e gráficos para comparar priors e posteriores.
    *   Gráficos de PPC (histogramas de dados observados vs. preditos).
    *   Matriz de correlação entre os parâmetros inferidos.
    *   Input para dados de validação destrutiva e exibição de comparação.
    *   Discussão sobre identificabilidade e recomendações para experimentos futuros.

Exemplos de Uso

A aplicação é projetada para ser explorada interativamente. Aqui estão alguns cenários de uso:

1.  Exploração do Modelo Direto:
    *   No Módulo 1, ajuste as constantes elásticas de um compósito típico (e.g., fibra de carbono/epóxi).
    *   Observe como as velocidades de onda variam com a direção de propagação (θ e φ).
    *   Altere uma constante de acoplamento (e.g., C₁₂) e veja seu impacto nas velocidades oblíquas.

2.  Simulação de Medição e Incerteza:
    *   No Módulo 2, simule um sinal ultrassônico para um material com propriedades conhecidas.
    *   Adicione ruído e atenuação.
    *   Observe como a qualidade do sinal afeta a extração do TOF e a incerteza na velocidade.

3.  Inferência de Parâmetros:
    *   No Módulo 3, insira um conjunto de velocidades experimentais (pode ser do Módulo 2 ou dados reais).
    *   Defina priors razoáveis para as 9 constantes elásticas e a densidade.
    *   No Módulo 4, execute o MCMC. Monitore os trace plots e diagnósticos para garantir a convergência.
    *   Analise os histogramas posteriores para obter as estimativas (média, desvio padrão, IC 95%) e os gráficos de correlação.

4.  Validação e Robustez:
    *   No Módulo 5, use as amostras MCMC do Módulo 4.
    *   Verifique a identificabilidade de cada constante. Quais são bem determinadas? Quais permanecem incertas?
    *   Execute o PPC para ver se o modelo Bayesiano é um bom preditor dos dados observados.
    *   Se tiver dados de ensaios destrutivos, insira-os para comparar com as estimativas Bayesianas.

Interpretação de Resultados

*   Trace Plots (Módulo 4): Devem parecer "lagartas difusas" sem tendências ou saltos abruptos, indicando que a cadeia explorou bem o espaço e atingiu a estacionaridade.
*   R-hat (Módulo 4): Valores próximos a 1.0 (idealmente < 1.05) para todos os parâmetros indicam que as múltiplas cadeias convergiram para a mesma distribuição posterior.
*   ESS (Módulo 4): Um valor alto (e.g., > 400 por parâmetro) indica que você tem amostras efetivamente independentes suficientes para estimativas confiáveis.
*   Histogramas Posteriores (Módulo 4): Representam a distribuição de probabilidade de cada parâmetro. A média é a estimativa pontual, e o desvio padrão (SD) ou Intervalos de Credibilidade (IC 95%) quantificam a incerteza.
*   Gráficos de Correlação (Módulo 4): Mostram como os parâmetros estão relacionados. Correlações fortes (próximas a +1 ou -1) podem indicar problemas de identificabilidade ou a necessidade de mais dados.
*   Razão SD Posterior/Prior (Módulo 5): Uma razão significativamente menor que 1 indica que os dados foram informativos para aquele parâmetro. Se a razão for próxima de 1, o parâmetro é mal-identificado pelos dados.
*   PPC (Módulo 5): Se os dados observados caem dentro da distribuição dos dados preditos pelo modelo, isso sugere que o modelo é adequado.

Troubleshooting

*   streamlit command not found: Certifique-se de que o ambiente virtual está ativado e que o Streamlit foi instalado corretamente (pip install streamlit).
*   ModuleNotFoundError: Verifique se todas as dependências listadas em requirements.txt foram instaladas (pip install -r requirements.txt).
*   MCMC não converge (R-hat alto, ESS baixo):
    *   Aumente o número de iterações e/ou o burn-in.
    *   Ajuste a escala da distribuição de proposta (se a taxa de aceitação for muito alta ou muito baixa).
    *   Verifique se os priors são muito restritivos ou se há inconsistência entre os priors e os dados.
    *   Pode indicar que o modelo é mal-especificado ou que os dados não são informativos o suficiente para alguns parâmetros.
*   Erros de memória: Reduza o número de iterações MCMC, o número de cadeias ou o tamanho do "thinning".
*   Resultados não-físicos: Verifique os limites dos seus priors. A inferência Bayesiana tende a respeitar os priors.

Referências Bibliográficas

As implementações exatas e os conceitos teóricos são baseados em literatura científica consolidada:

*   Bayesian Inference:
       Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis*. Chapman and Hall/CRC.
       Tarantola, A. (2005). Inverse Problem Theory and Methods for Model Parameter Estimation*. SIAM.
*   Ultrasonics & Anisotropic Elasticity:
       Truell, R., Elbaum, C., & Chick, B. B. (1969). Ultrasonic Methods in Solid State Physics*. Academic Press.
       Auld, B. A. (1990). Acoustic Fields and Waves in Solids*. Krieger Publishing Company.
*   Composite Mechanics:
       Jones, R. M. (1999). Mechanics of Composite Materials*. Taylor & Francis.
*   MCMC & Diagnostics:
       Brooks, S., Gelman, A., Jones, G., & Meng, X. L. (Eds.). (2011). Handbook of Markov Chain Monte Carlo*. Chapman and Hall/CRC.
       Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation using multiple sequences. Statistical Science*, 7(4), 457-472.

Contribuições

Contribuições são bem-vindas! Se você encontrar um bug, tiver uma sugestão de melhoria ou quiser adicionar uma nova funcionalidade, sinta-se à vontade para:

1.  Abrir uma issue no repositório.
2.  Fazer um fork do projeto e enviar um pull request.

Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.
`