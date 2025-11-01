# Food Recognition with Teachable Machine and TensorFlow

## Objetivo

Este projeto utiliza um modelo exportado do **Teachable Machine** para identificar alimentos em tempo real a partir da webcam, mostrando o nome do alimento, a confiança da predição e informações nutricionais (calorias, proteínas, carboidratos e gorduras) quando a confiança é igual ou superior a 90%.

## Dependências

- Python 3.10+
- TensorFlow 2.x
- OpenCV
- NumPy

Instalação das dependências:

```bash
pip install -r requirements.txt
```

## Estrutura do projeto

fitaura.iot/  
│  
├─ dataset/  
│ └─ treino.zip  
├─ model/  
├─ model.savedmodel/  
│ ├─ assets/  
│ ├─ variables/  
│ │ ├─ variables.data-00000-of-00001  
│ │ └─ variables.index  
│ └─ saved_model.pb  
│  
├─ labels.txt  
├─ .gitignore  
├─ main.py  
├─ README.md  
└─ requirements.txt

## Descrição dos Arquivos e Diretórios

- **`dataset/treino.zip`** - Dataset de treino compactado
- **`model/`** - Diretório do modelo
- **`model.savedmodel/`** - Modelo salvo do TensorFlow
  - **`assets/`** - Arquivos auxiliares do modelo
  - **`variables/`** - Variáveis do modelo treinado
  - **`saved_model.pb`** - Arquivo principal do modelo
- **`labels.txt`** - Labels/classes do modelo
- **`.gitignore`** - Arquivo de exclusão para Git
- **`main.py`** - Script principal da aplicação
- **`README.md`** - Documentação do projeto
- **`requirements.txt`** - Dependências do projeto

## Execução

Para rodar o programa:

```bash
python main.py
```

A aplicação abrirá a webcam e exibirá uma janela com as predições em tempo real. Pressione Q para sair.

## Parâmetros

- Confiança mínima para exibir informações: 90%
- Redimensionamento da imagem: 224x224 pixels
- Normalização da imagem: img = (img / 127.5) - 1.0

## Membros

**Gabriel Leão da Silva** - RM 552642  
**Matheus Farias de Lima** - RM 554254  
**Miguel Mauricio Parrado Patarroyo** - RM 554007  
**Vitor Pinheiro Nascimento** - RM 553693
