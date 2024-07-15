import pandas as pd
import os


def ler_planilha(file_name):
    file_path = os.path.join(os.path.dirname(__file__), '../../resources', file_name)

    # Normalizar o caminho para garantir compatibilidade com diferentes sistemas operacionais
    file_path = os.path.normpath(file_path)

    # Ler a planilha (a primeira aba) e definir a primeira linha como cabeçalho
    df = pd.read_excel(file_path, header=0)

    return df


def remover_colunas_escala(df):
    escalas_textuais = {'DISCORDO', 'CONCORDO', 'DISCORDO TOTALMENTE', 'CONCORDO TOTALMENTE',
                        'NEM DISCORDO, NEM CONCORDO','SIM'}
    escalas_numericas = {'1', '2', '3', '4', '5'}

    colunas_para_remover = []

    for coluna in df.columns:
        if 'NOME' in coluna.strip().upper():
            colunas_para_remover.append(coluna)
            continue

        contem_escalas_ou_arroba_ou_timestamp = False

        for value in df[coluna].dropna().astype(str):
            value_upper = value.strip().upper()
            if (
                    value_upper in escalas_textuais or value_upper in escalas_numericas or '@' in value_upper or pd.to_datetime(
                    value, errors='coerce') is not pd.NaT):
                contem_escalas_ou_arroba_ou_timestamp = True
                break

        if contem_escalas_ou_arroba_ou_timestamp:
            colunas_para_remover.append(coluna)

    df_filtrado = df.drop(columns=colunas_para_remover)

    return df_filtrado



