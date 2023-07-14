# Fluxo de Active Learning para modelos de Reconhecimento de entidades nomeadas (NER)

Esse repositório possui códigos fonte do trabalho de conclusão de curso de Engenharia de Computação do aluno Matheus Gabriel, do curso de Engenharia de Software, intitulado "Construção de uma pipeline de aprendizagem ativa para modelos de reconhecimento de entidades nomeadas".

## Como executar

```bash
# Crie a network para o docker
$ docker network create active_learning

# Crie o volume para o banco de dados do label-studio
$ docker volume create pgdata

# Crie o arquivo .env
$ cp .env.example .env

# Execute o docker
$ docker-compose up -d
```

Em seguida, acesse o Label Studio em [http://localhost:8508](http://localhost:8508), crie uma conta e copie a chave de `Access token` localizada em:

 `Account & Settings >  Access Token`  
 
 cole o valor da chave no arquivo .env na variável `LABEL_STUDIO_ACCESS_TOKEN`, salve o arquivo e reinicie o docker com o seguinte comando:

```bash
docker-compose stop backend && docker-compose up -d backend 
```
