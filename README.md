Link para dataset: https://www.dropbox.com/scl/fo/zgtiryigax107nddldshi/AO2tYU4Dh0KyJ-f8pKOmpEY?rlkey=sf91fcnznh73w97z72skmrybe&dl=0
folder com as imagens já registradas: openSlide_level_4_To_level_0_rigid_reg



# Predição de pCR em TNBC com coloração virtual

Repositório da dissertação sobre predição de resposta patológica completa (pCR) em câncer de mama triplo-negativo (TNBC), combinando:
- imagens H&E,
- colorações Ki-67 e PHH3 (reais ou virtuais),
- atenção espacial baseada em biomarcadores.

A abordagem é inspirada em Duanmu et al. (2022), removendo o módulo de detecção de células tumorais e incorporando um módulo de coloração virtual (GANs).

## Estrutura (resumo)

- `bcs-kedro/`  
Projeto Kedro contendo o pipeline completo de treinamento e avaliação para predição de pCR:
- Pré-processamento de imagens
- Geração de mapas de atenção espacial baseados em biomarcadores
- Modelo de predição baseado em ResNet com atenção

- `virtual_staining/`  
Implementação dos modelos de coloração virtual:
- **CycleGAN**: tradução imagem-para-imagem não pareada
- **cGAN**: GAN condicional
- **Difusão**: modelos baseados em Difusão

Inclui scripts para treinar e gerar colorações virtuais de Ki-67 e PHH3 a partir de H&E.


## 🚀 Instalação e Configuração

### 1. Clonar o repositório

git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

### 2. Criar ambiente virtual

Recomenda-se Python 3.8 ou superior:

python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

### 3. Instalar dependências

pip install -r requirements.txt

---

## ▶️ Execução

### Pipeline de Predição de pCR (Kedro)

1. **Navegue até o diretório do projeto Kedro:**

cd bcs-kedro

Rode o arquivo .sh localizado em bcs-kedro/scripts/run_train_cnn_pcr.sh  alterando somente o pipeline e o seu user que deseja executar, como: 
username="henrique"
kedro_pipeline="train_cnn"

e configurações de execução, como górgona desejada, tempo de alocação e nome do job, exemplo:
#SBATCH --job-name=trainCNNPCR_9_channels
#SBATCH --time=10:00:00
#SBATCH --nodes=1 
#SBATCH --nodelist=gorgona7
#SBATCH --output=(path de output)/slurm-logs/slurm-%j.log

os pipelines disponíveis estão em: bcs-kedro/src/bcs_kedro/pipelines , cada folder é um pipeline existente.



### Coloração Virtual

1. **Navegue até o diretório de coloração virtual:**

cd virtual_stainning

Rode o arquivo .sh, alterando somente o .py que deseja executar, e configurações de execução, como górgona desejada, tempo de alocação e nome do job.

## 🔧 Configuração Adicional

### Ajustes Importantes

- **Caminhos de dados**: Certifique-se de que os caminhos nos arquivos de configuração (`conf/base/catalog.yml` no Kedro, e scripts `.sh` no módulo de coloração virtual) apontam corretamente para seus dados.

- **GPU/CPU**: O treinamento é otimizado para GPU. Se estiver usando CPU, ajuste os batch sizes nos scripts de treinamento.
