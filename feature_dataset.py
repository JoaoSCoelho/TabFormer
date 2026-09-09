import pandas as pd
import numpy as np

def adicionar_features_movimento(caminho_arquivo):
  """Carrega um arquivo CSV, calcula as razões de movimento

  e salva com '_FEATURED' no final do nome.
  """
  # 1. Carregar o arquivo CSV
  df = pd.read_csv(caminho_arquivo)

  # 2. Criar as novas colunas de proporção/razão
  # Usamos .replace(0, float('nan')) opcionalmente para evitar divisões por zero,
  # mas a divisão direta por padrão gerará inf/nan caso haja zeros.
  df['journal_entry_movement_ratio'] = (
      df['movement'] / df['journal_entry_movement']
  ).replace([np.inf, -np.inf], 1)

  df['global_movement_ratio'] = (df['movement'] / df['global_movement']).replace(
      [np.inf, -np.inf], 1
  )

  # 3. Definir o nome do arquivo de saída (substitui '.csv' por '_FEATURED.csv')
  caminho_saida = caminho_arquivo.replace('.csv', '_FEATURED.csv')

  # 4. Salvar o novo arquivo CSV (index=False evita salvar o índice do pandas)
  df.to_csv(caminho_saida, index=False)

  print(f'Processo concluído! Arquivo salvo em: {caminho_saida}')



adicionar_features_movimento("Base_Contábil___Projeto_NuFuturo_anonimizado.csv")