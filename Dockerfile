# Use uma imagem base do Python
FROM python:3.9.13

# Atualize pip, setuptools, e wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Defina o diretório de trabalho dentro do container
WORKDIR app

# Copie o arquivo de dependências (requirements.txt) para o container
COPY requirements.txt .

# Instale as dependências necessárias
RUN pip install --no-cache-dir -r requirements.txt

# Copie o resto do código para o diretório de trabalho no container
COPY . .

# Defina o comando padrão para rodar o código
CMD ["python", "src/model/train.py"]