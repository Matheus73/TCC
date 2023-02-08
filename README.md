# Rotulador

Esse repositório possui códigos para executar a tarefa exportação e importação no Label Studio

## Como executar

```bash
# Clone este repositório
$ git clone https://gitlab.com/gpam/osiris/research/rotulador.git

# Acesse a pasta do projeto no terminal/cmd
$ cd rotulador

# Crie a network para o docker
$ docker network create rotulador

# Crie o volume para o banco de dados do label-studio
$ docker volume create pgdata

# Crie o arquivo .env
$ cp .env.example .env

# Execute o docker
$ docker-compose up -d
```

Em seguida, acesse o Label Studio em http://localhost:8508, crie uma conta e copie a chave de `Access token` localizada em `Account & Settings >  Access Token`  cole o valor da chave no arquivo .env na variável `LABEL_STUDIO_ACCESS_TOKEN`, salve o arquivo e reinicie o docker com o seguinte comando:

```bash
docker-compose stop backend && docker-compose up -d backend 
```
