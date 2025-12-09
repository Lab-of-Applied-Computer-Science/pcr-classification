# Predição de pCR em TNBC com coloração virtual

Repositório da dissertação sobre predição de resposta patológica completa (pCR) em câncer de mama triplo-negativo (TNBC), combinando:
- imagens H&E,
- colorações Ki-67 e PHH3 (reais ou virtuais),
- atenção espacial baseada em biomarcadores.

A abordagem é inspirada em Duanmu et al. (2022), removendo o módulo de detecção de células tumorais e incorporando um módulo de coloração virtual (GANs).

## Estrutura (resumo)

- `bcs-kedro/`  
  Projeto Kedro com o pipeline de treinamento e avaliação para predição de pCR
  (pré-processamento, geração de mapas de atenção e modelo ResNet).

- `virtual_stainning/`  
  Implementação dos modelos de coloração virtual (CycleGAN, cGAN, difusão) e
  scripts para treinar e gerar Ki-67/PHH3 virtuais a partir de H&E.

- requirements.txt utilizado para criação de env


