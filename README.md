# Bayesiana
Inferência Bayesiana para caracterização de compósitos

---

🔬 Bayesian Inference for Composite Characterization via Ultrasound

🚀 Visão Geral

Esta aplicação Streamlit é uma ferramenta interativa e educacional projetada para explorar e aplicar os princípios da Inferência Bayesiana na caracterização de propriedades elásticas de laminados compósitos usando ultrassom. Ela consolida cinco módulos distintos, cobrindo desde os fundamentos da propagação de ondas em materiais anisotrópicos até a validação de modelos Bayesianos complexos.

A aplicação permite que usuários, desde estudantes a pesquisadores e engenheiros, visualizem e interajam com cada etapa do processo de inferência, compreendendo como dados experimentais de ultrassom podem ser usados para estimar parâmetros materiais e quantificar suas incertezas.

✨ Funcionalidades Principais

*   Módulo 1: Fundamentos - Explore a equação de Christoffel e a propagação de ondas em meios anisotrópicos.
*   Módulo 2: Medição por Ultrassom - Simule a extração de velocidades e a propagação de incertezas em medições ultrassônicas.
*   Módulo 3: Inferência Bayesiana - Entenda a construção da função de verossimilhança (likelihood) e a importância dos priors na formação da posterior.
*   Módulo 4: MCMC - Execute simulações de Markov Chain Monte Carlo (MCMC) usando o algoritmo Metropolis-Hastings e analise seus diagnósticos de convergência.
*   Módulo 5: Validação e Análise de Sensibilidade - Avalie a identificabilidade dos parâmetros, a sensibilidade ao prior e a adequação do modelo através de análises preditivas posteriores (PPC).

💻 Requisitos do Sistema

*   Sistema Operacional: Windows, macOS, Linux (compatível com Python).
*   Python: Versão 3.8 ou superior.
*   Memória RAM: Mínimo de 4GB (8GB ou mais recomendado para simulações MCMC mais longas).
*   CPU: Processador multi-core recomendado para melhor desempenho.

📦 Instalação

Siga os passos abaixo para configurar e executar a aplicação em seu ambiente local.

1. Pré-requisitos

Certifique-se de ter o Python instalado em seu sistema. Você pode baixá-lo em python.org.

2. Criar e Ativar um Ambiente Virtual (Recomendado)

É uma boa prática isolar as dependências do projeto em um ambiente virtual.

`bash
Crie o ambiente virtual
python -m venv venv

Ative o ambiente virtual
No Windows:
.\venv\Scripts\activate
No macOS/Linux:
source venv/bin/activate
`

3. Baixar o Código-Fonte

Assumindo que você tem o arquivo main_app.py e requirements.txt no mesmo diretório:

`bash
Se você clonar um repositório (ex: git clone <URL_DO_REPOSITORIO>)
Ou simplesmente coloque os arquivos main_app.py e requirements.txt em uma pasta
cd /caminho/para/sua/pasta/do/projeto
`

4. Instalar as Dependências

Com o ambiente virtual ativado, instale todas as bibliotecas necessárias usando o requirements.txt fornecido:

`bash
pip install -r requirements.txt
`

▶️ Como Executar a Aplicação

Após a instalação das dependências, execute a aplicação Streamlit:

`bash
streamlit run main_app.py
`

Isso abrirá automaticamente a aplicação em seu navegador padrão (geralmente em http://localhost:8000 ou uma porta similar).

📖 Descrição dos Módulos e Uso

A navegação entre os módulos é feita através de um menu na barra lateral esquerda da aplicação Streamlit.

1. Módulo: Fundamentos (Christoffel Solver)

*   Objetivo: Visualizar como as propriedades elásticas (C_ij) e a densidade (ρ) afetam as velocidades de propagação de ondas em diferentes direções.
*   Controles: Sliders para ajustar C₁₁, C₁₂, C₃₃, C₄₄, C₅₅, C₆₆ e ρ. Seletores para a direção de propagação (ângulo).
*   Saída: Exibição da matriz de Christoffel, autovalores (ρv²), velocidades de fase e polarizações para os três modos de onda. Gráficos polares das velocidades.

2. Módulo: Medição por Ultrassom (TOF & Incertezas)

*   Objetivo: Simular uma medição ultrassônica e analisar a propagação de incertezas.
*   Controles: Entradas numéricas para espessura da amostra (h), TOF medido, e suas respectivas incertezas (δh, δTOF).
*   Saída: Cálculo da velocidade da onda (v), incerteza relativa em v, e uma visualização da distribuição de probabilidade da velocidade.

3. Módulo: Inferência Bayesiana (Likelihood, Prior, Posterior)

*   Objetivo: Entender os componentes da inferência Bayesiana: likelihood, prior e posterior.
*   Controles: Sliders para ajustar valores hipotéticos de C₁₁ e ρ. Seletores para o tipo de prior (Uniforme, Gaussiano).
*   Saída: Visualização da função de likelihood para um dado conjunto de medições, da distribuição prior e da distribuição posterior (conceitual).

4. Módulo: MCMC (Metropolis-Hastings & Diagnósticos)

*   Objetivo: Executar uma simulação MCMC para estimar parâmetros e diagnosticar a convergência.
*   Controles: Entradas para número de iterações, tamanho do passo da proposta (proposal step size), e valores iniciais para os parâmetros.
*   Saída: Gráficos de traço (trace plots) para cada parâmetro, histogramas das distribuições posteriores, cálculo do R-hat (Gelman-Rubin) e Effective Sample Size (ESS).

5. Módulo: Validação e Análise de Sensibilidade

*   Objetivo: Avaliar a robustez e confiabilidade dos resultados da inferência Bayesiana.
*   Controles: Seletores para diferentes configurações de prior (para análise de sensibilidade), e opções para gerar dados sintéticos para Posterior Predictive Check (PPC).
*   Saída:
    *   Identificabilidade: Tabela comparando SD do prior vs. SD do posterior para cada parâmetro.
    *   Correlações: Matriz de correlação entre os parâmetros posteriores.
    *   PPC: Gráficos comparando dados observados com dados simulados a partir da posterior.
    *   σ_v: Comparação do tratamento de incertezas na likelihood Bayesiana vs. least-squares clássico.

💡 Exemplos de Uso

*   Explorar Anisotropia: No Módulo 1, ajuste os ângulos de propagação e observe como as velocidades e polarizações mudam drasticamente em um material compósito.
*   Impacto da Incerteza: No Módulo 2, aumente a incerteza na espessura ou no TOF e veja como a incerteza na velocidade calculada se propaga.
*   Entender o Prior: No Módulo 3, mude a largura de um prior uniforme e observe como isso afeta a forma da posterior, especialmente se a likelihood for fraca.
*   Diagnosticar MCMC: No Módulo 4, experimente com um proposal step size muito pequeno ou muito grande e observe como o trace plot e o R-hat indicam má convergência.
*   Identificar Parâmetros Fracos: No Módulo 5, observe a razão SD_prior/SD_posterior. Se for próxima de 1, o parâmetro é mal-identificado pelos dados atuais.

⚠️ Troubleshooting

*   streamlit command not found: Certifique-se de que o Streamlit está instalado (pip install streamlit) e que seu ambiente virtual está ativado.
*   ModuleNotFoundError: Verifique se todas as dependências listadas em requirements.txt foram instaladas corretamente (pip install -r requirements.txt).
*   Application not loading in browser: Verifique o console onde você executou streamlit run main_app.py para ver o endereço IP e a porta. Pode haver um problema de firewall bloqueando a porta.
*   Slow performance: Simulações MCMC podem ser computacionalmente intensivas. Reduza o número de iterações ou o número de parâmetros para testes iniciais.
*   Errors in the Streamlit app: O Streamlit geralmente exibe mensagens de erro úteis diretamente na interface. Verifique o console para rastreamentos de pilha mais detalhados.

📚 Referências Teóricas

Esta aplicação é baseada em conceitos de:

*   Mecânica dos Materiais Compósitos: Teoria da elasticidade para materiais anisotrópicos, matrizes de rigidez.
*   Propagação de Ondas Ultrassônicas: Equação de Christoffel, modos de onda, extração de tempo de voo.
*   Estatística Bayesiana: Teorema de Bayes, funções de verossimilhança, distribuições prior e posterior.
*   Métodos de Monte Carlo via Cadeias de Markov (MCMC): Algoritmo Metropolis-Hastings, diagnósticos de convergência (R-hat, ESS).
*   Análise de Incertezas: Propagação de erros, validação de modelos.

Livros Recomendados:

   Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis*. Chapman and Hall/CRC.
   Jones, R. M. (1999). Mechanics of Composite Materials*. CRC Press.
   Truell, R., Elbaum, C., & Chick, B. B. (1969). Ultrasonic Methods in Solid State Physics*. Academic Press.

---

Desenvolvido por: [Seu Nome/Organização, se desejar]
Data: [Data da criação/última atualização]
`

