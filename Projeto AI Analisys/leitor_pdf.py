import pdfplumber
import tabula
# Função para juntar o texto de células da tabela com espaços
def extract_text_with_spaces(row, column_index):
    cell_text = row[column_index]
    if cell_text is not None:
        # Substitui múltiplos espaços por um único espaço e remove espaços extras nas bordas
        return ' '.join(cell_text.split())
    return ''

lista_tabelas = tabula.read_pdf("TESTE.pdf", pages=1, encoding="utf-8")

# Abre o arquivo PDF
with pdfplumber.open('Modelo Questionário - ATLAS - Canva.pdf') as pdf:
    # Seleciona a página (se o PDF tiver mais de uma página, ajuste conforme necessário)
    page = pdf.pages[0]
    
    # Extrai tabelas da página
    tables = page.extract_tables()

    # Verifica se há tabelas extraídas
    if lista_tabelas:
        # Supondo que a primeira tabela seja a que você deseja
        lista_tabelas = lista_tabelas[0]
        
        # Itera sobre as linhas da tabela e imprime a segunda coluna
        for row in lista_tabelas:
            # Verifica se a linha tem pelo menos duas colunas
            if len(row) > 1:
                 # Extrai o texto da segunda coluna da linha, garantindo que os espaços sejam preservados
                    text = extract_text_with_spaces(row, 1)
                    print(text)
    else:
        print("Nenhuma tabela encontrada na página.")
