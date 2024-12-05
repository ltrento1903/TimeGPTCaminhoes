import numpy as np
import pandas as pd
from utilsforecast.evaluation import evaluate
from utilsforecast.plotting import plot_series
from utilsforecast.losses import mse, mae, rmse, smape, bias
from nixtla import NixtlaClient
import streamlit as st

st.set_page_config(page_title='NIXTLA TimeGPT - NO PARADIGMA FORECAST', 
                   layout='wide', page_icon='🎢', initial_sidebar_state='auto')

st.title('TimeGPT: A Revolução da Previsão')


st.image('https://th.bing.com/th/id/OIP.3gbJuoIfIp_g7z4ER6d7oAHaEK?rs=1&pid=ImgDetMain', width=800)


st.markdown("Saiba mais sobre o TimeGPT no [site oficial](https://docs.nixtla.io/).")

st.markdown('''***Otimizando Previsões de Demanda de Caminhões com TimeGPT: Um Experimento de Fine-tuning***

Investigamos o impacto de diferentes estratégias de fine-tuning na precisão das previsões geradas pelo TimeGPT, utilizando como caso de estudo a demanda por caminhões nas categorias semileves, leves, médios, semipesados e pesados. As abordagens avaliadas incluem:

1. Modelo Base: TimeGPT pré-treinado.
2. Fine-tuning Padrão: Ajuste fino do modelo aos dados específicos.
3. Fine-tuning com MAE: Minimização do erro absoluto médio para previsões mais robustas.
4. Fine-tuning Avançado: Combinação de MAE e ajuste da profundidade das camadas para maior refinamento.
Os resultados revelam que o fine-tuning, especialmente quando personalizado com a métrica MAE e ajustes na arquitetura do modelo, oferece previsões mais precisas e confiáveis, demonstrando o potencial do TimeGPT como ferramenta para a análise de séries temporais complexas."
''')

col1, col2=st.columns([3,1.5], gap='large')

with col1:

    st.write('***Gráfico Previsâo 1. Modelo Base***')
    st.image(r'C:\Users\luizt\my_streamlit_app\forecast1.png', width=1400)

with col2:

    st.markdown('''O gráfico mostra as séries temporais e as previsões geradas pelo modelo ***TimeGPT*** para diferentes categorias de ***caminhões: Caminhões, Semileves, Leves, Médios, Semipesados, e Pesados***. Aqui está uma análise detalhada:
________________________________________
1. Comportamento Geral
•	Dados Históricos (linha azul): Os dados mostram padrões variados entre as categorias:
o	Caminhões e Pesados: Tendência geral de crescimento após períodos de queda ou estabilização.
o	Semileves e Leves: Declínio contínuo ou estabilização em níveis baixos.
o	Médios e Semipesados: Apresentam variações cíclicas e crescimento recente.
•	Previsões (linha azul clara): O modelo captura bem as tendências recentes para todas as categorias, ajustando-se aos últimos padrões históricos antes da previsão.
________________________________________
2. Intervalos de Confiança
•	Áreas sombreadas (90% e 95%): 
o	São mais amplas para horizontes de previsão mais longos, o que reflete maior incerteza futura.
o	Caminhões e Pesados: Incertezas são moderadas, indicando previsões confiáveis.
o	Semileves e Leves: Intervalos amplos, sugerindo maior incerteza devido à volatilidade ou volume de dados reduzido.
o	Semipesados e Médios: Variabilidade capturada, mas com confiança aceitável.
________________________________________
3. Tendências e Padrões
•	Caminhões: Retomada de crescimento consistente, capturada adequadamente pelo modelo.
•	Semileves e Leves: O modelo prevê estabilidade com pouca variação.
•	Médios e Semipesados: Crescimento contínuo previsto, com sazonalidade moderada captada.
•	Pesados: Forte tendência de crescimento, bem modelada com intervalos de confiança ajustados.
________________________________________
4. Possíveis Problemas ou Melhorias
•	Semileves e Leves: A incerteza alta sugere que o modelo pode estar encontrando dificuldade em capturar padrões claros devido à baixa variabilidade ou dados insuficientes.
•	Semipesados: Embora as previsões estejam alinhadas à tendência, os intervalos são mais amplos, sugerindo possível ruído nos dados.
________________________________________
5. Ações Recomendadas
•	Validação: Avaliar a precisão das previsões utilizando dados futuros ou um conjunto de teste.
•	Análise de Sazonalidade: Explorar variáveis sazonais ou exógenas (como economia, regulamentações) para refinar o modelo.
•	Segmentação: Caso categorias menores apresentem alta incerteza, pode ser útil agregá-las a séries maiores (ex.: juntar "Leves" e "Semileves").
________________________________________
No geral, o ***TimeGPT*** demonstrou bom desempenho na captura das tendências das séries temporais. Ajustes finos podem melhorar as previsões em categorias com maior incerteza. 
''')


# Configuração do cliente da Nixtla
    nixtla_client = NixtlaClient(
        api_key='nixak-vgekzclAmNcdPsAOSfaQ7WcDXWgGjU6X5peo92QoKCUKG4wDFq3AO1yU6GXStQRyGPHVw21Zao78DknY'
    )

col1, col2, col3=st.columns([2,2,2], gap='large')

with col1:

# Carregar os dados
    df = pd.read_excel(r"C:\Tablets\Caminhões_analises.xlsx")  
    st.write('***Dataframe Original: Base de dados Anfavea***', height=300, column_config=dict)
    st.dataframe(df)

    # Renomear a coluna de datas para 'ds' e garantir o formato correto
    df = df.rename(columns={"Mês": "ds"})
    df['ds'] = pd.to_datetime(df['ds'])



    # Reestruturar os dados para o formato esperado pelo StatsForecast
    series_cols = ['Caminhões', 'Semileves', 'Leves', 'Médios', 'Semipesados', 'Pesados']
    df_long = df.melt(id_vars='ds', value_vars=series_cols, var_name='unique_id', value_name='y')

    st.write('***Dataframe ajustado para o padrão TimeGPT***')
    st.dataframe(df_long, height=300)

# Exibição inicial do DataFrame
    print("Visualização inicial dos dados:")
    print(df_long.head())

    df_sub = df_long.query('unique_id == "DE"')


    df_train = df_long.query('ds < "2022-05-01"')
    df_test = df_long.query('ds >= "2022-05-01"')    
       
      

    plot_series(df_train[['unique_id','ds','y']][-200:], forecasts_df= df_test[['unique_id','ds','y']].rename(columns={'y': 'test'}))
    fcst_timegpt = nixtla_client.forecast(df = df_train[['unique_id','ds','y']],
                                        h=2*24,
                                        target_col = 'y',
                                        level = [90, 95])

with col2:
    st.write('***Previsão de Demanda  1. Modelo Base***')
    st.dataframe(fcst_timegpt, height=300)

    metrics = [mae, rmse, smape, bias]

    evaluation = evaluate(
        fcst_timegpt.merge(df_test, on=['unique_id', 'ds']),
        metrics=metrics,
        models=['TimeGPT']
    )

    
    evaluation.to_excel('C:\Tablets/evaluation.xlsx', index=False)

    print(evaluation)

    st.write('***Métricas de Performance 1. Modelo Base***')
    st.dataframe(evaluation)

# Plotar séries com previsões
    plot_series(
        df_long,  # Série histórica
        forecasts_df=fcst_timegpt,  # Previsões
        level=[90, 95]  # Intervalos de confiança
    )
    
with col3:

    fcst_finetune_df = nixtla_client.forecast(df=df_train[['unique_id', 'ds', 'y']],
                                            h=24*2,
                                            finetune_steps = 50,
                                            level=[90, 95])

    st.write('***Previsão de Demanda  2.Fine-tuning Padrão: finetune_steps=50***')
    st.dataframe(fcst_finetune_df, height=300)

    evaluation_1 = evaluate(
        fcst_finetune_df.merge(df_test, on=['unique_id', 'ds']),
        metrics=metrics,
        models=['TimeGPT']
    )
    print(evaluation_1)
    
    st.write('***Métricas de Performance 2.Fine-tuning Padrão***')
    st.dataframe(evaluation_1, height=300)

    # Plotar séries com previsões
    plot_series(
        df_long,  # Série histórica
        forecasts_df=fcst_finetune_df,  # Previsões
        level=[90, 95]  # Intervalos de confiança
    )

col1, col2=st.columns([1,1], gap='large')

with col1:

    st.write('***Previsão de Demanda  3. Fine-tuning (finetune_steps=50) + finetune_loss = mae***')
    
    fcst_finetune_mae_df = nixtla_client.forecast(df=df_train[['unique_id', 'ds', 'y']],
                                            h=24*2,
                                            finetune_steps = 50,
                                            finetune_loss = 'mae',
                                            level=[90, 95])

    st.dataframe(fcst_finetune_mae_df, height=300)

    evaluation_2 = evaluate(
        fcst_finetune_mae_df.merge(df_test, on=['unique_id', 'ds']),
        metrics=metrics,
        models=['TimeGPT']
    )
    print(evaluation_2)
    
    st.write('***Métricas de Performance 3. Fine-tuning com MAE***')
    st.dataframe(evaluation_2, height=300)

    # Plotar séries com previsões
    plot_series(
        df_long,  # Série histórica
        forecasts_df=fcst_finetune_mae_df,  # Previsões
        level=[90, 95]  # Intervalos de confiança
    )

with col2:

    st.write('***Previsão de Demanda  4. Fine-tuning Avançado: finetune_steps = finetune_steps= 50, finetune_depth=1, finetune_loss= mae***')
    
    fcst_finetune_depth_df = nixtla_client.forecast(df=df_train[['unique_id', 'ds', 'y']],
                                                    h=24*2,
                                                    finetune_steps = 50,
                                                    finetune_depth=1,
                                                    finetune_loss = 'mae',
                                                    level=[90, 95])

    st.dataframe(fcst_finetune_depth_df, height=300)

    evaluation_3 = evaluate(
        fcst_finetune_depth_df.merge(df_test, on=['unique_id', 'ds']),
        metrics=metrics,
        models=['TimeGPT']
    )
    print(evaluation_3)
    
    st.write('***Métricas de Performance 4. Fine-tuning Avançado***')
    st.dataframe(evaluation_3, height=300)

    # Plotar séries com previsões
    plot_series(
        df_long,  # Série histórica
        forecasts_df=fcst_finetune_depth_df,  # Previsões
        level=[90, 95]  # Intervalos de confiança
    )

col1, col2=st.columns([4,1], gap='large')

with col1:

    df_evaluation = pd.read_excel(r"C:\Tablets\df_evaluation.xlsx")  
    st.header('Compilado das Quatro Análises de ajustes no Modelo')
    st.markdown('''### ***Dataframe das Quatro Análises de Ajustes no Modelo*** 

    Este compilado apresenta uma visão consolidada dos ajustes feitos no modelo, destacando seus impactos sobre as categorias analisadas e as principais métricas: MAE, RMSE, SMAPE e Bias.
''')
    st.write('''***Dataframe Medidas de Performance***''')
    st.dataframe(df_evaluation, hide_index=300)

with col2: 

    st.markdown('''### ***Análises das Quatro Análises de Ajustes no Modelo*** 

---

    A tabela apresenta métricas para previsões em diferentes categorias de veículos ("Caminhões", "Leves", etc.), utilizando diferentes métodos de ajuste fino ("fine-tuning"). Abaixo está uma análise dos resultados principais:

---

### **1. Análise Geral**
- **Métricas Avaliadas**: MAE (Erro Médio Absoluto), RMSE (Erro Quadrático Médio), SMAPE (Erro Percentual Médio Absoluto Simétrico) e Bias.
- **Objetivo**: Comparar o desempenho entre:
  1. **Modelo TimeGPT Original (`fcst_timegpt`)**
  2. **Fine-tuning padrão (50 steps)**
  3. **Fine-tuning com `finetune_loss = mae`**
  4. **Fine-tuning Avançado (`finetune_depth=1` e `finetune_loss=mae`)**
- **Variação**: A diferença entre os valores obtidos e os valores do modelo original.

---

### **2. Observações por Métrica**
#### **MAE (Erro Médio Absoluto)**
- Em quase todas as categorias (exceto "Pesados"), há uma redução significativa no erro após o fine-tuning.
- **Destaques Positivos**:
  - **"Caminhões"**: Redução de 88,91.
  - **"Leves"**: Redução de 29,91.
  - **"Médios"**: Redução expressiva de 82,79.
- **Exceção**: Em "Pesados", o MAE aumentou levemente (+25,96), indicando possível sobreajuste ou ineficácia no ajuste fino.

#### **RMSE (Erro Quadrático Médio)**
- Resultados similares ao MAE, com reduções marcantes na maioria das categorias:
  - **"Caminhões"**: Redução de 268,06.
  - **"Leves"**: Redução de 36,46.
  - **"Médios"**: Redução de 81,77.
- **"Pesados"**: Pequeno aumento de 5,76.

#### **SMAPE (Erro Percentual Médio Absoluto Simétrico)**
- Em geral, todos os métodos de ajuste fino reduziram os valores de SMAPE, exceto "Pesados" (aumento pequeno de +0,00268).
- **Melhores Reduções**:
  - **"Leves"**: Redução de 0,0168.
  - **"Médios"**: Redução de 0,0347.
  - **"Semipesados"**: Redução de 0,0056.

#### **Bias**
- O bias mostra uma redução expressiva para todas as categorias, indicando previsões mais equilibradas após o fine-tuning:
  - **"Caminhões"**: Redução massiva de 740,43.
  - **"Leves"**: Redução de 61,89.
  - **"Pesados"**: O maior ajuste, com redução de 428,57.
  - **"Semileves"**: Redução moderada de 12,08.

---

### **3. Comparação de Métodos**
#### **Fine-tuning Padrão vs. Fine-tuning Avançado**
- O método avançado (com `finetune_depth=1`) apresentou resultados iguais ao ajuste com `finetune_loss=mae`, sugerindo que a profundidade extra não trouxe ganhos adicionais significativos.
- Ambos os métodos são consistentemente melhores que o TimeGPT original, exceto na categoria "Pesados".

#### **Variação Geral**
- A maior redução em erro foi observada na categoria **"Caminhões"** (RMSE e Bias).
- **Categorias Críticas**:
  - **"Pesados"** apresentou aumentos no MAE e RMSE, o que requer análise adicional.

---

### **4. Conclusões**
- **Melhorias Consistentes**: O fine-tuning reduziu os erros (MAE, RMSE, SMAPE) na maioria das categorias, com destaque para "Caminhões" e "Leves".
- **Pontos de Atenção**:
  - A categoria "Pesados" apresentou aumentos leves em algumas métricas, indicando necessidade de ajustes nos parâmetros de treinamento.
- **Recomendações**:
  - Ajustar parâmetros ou dados de entrada para a categoria "Pesados".
  - Focar no fine-tuning avançado para obter maior estabilidade nas previsões.
    ''')



