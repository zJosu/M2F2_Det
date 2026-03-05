#!/usr/bin/env python3
"""Write the trimmed Deepfakes.tex report."""

content = r"""%% Informe técnico — Detección de Deepfakes con M2F2-Det
%% Autores: Josu Zabala Muxika, Javier Solana Crespo
%% Mondragon Unibertsitatea — Trustworthy AI (ML2)
%%
\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}

\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{mathrsfs}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\renewcommand{\tablename}{Tabla}
\renewcommand{\figurename}{Figura}

\raggedbottom

\begin{document}

\title[Detección Multimodal de Deepfakes con M2F2-Det]{Detección Multimodal de Deepfakes mediante Modelos Visión-Lenguaje: Aplicación y Fine-Tuning de M2F2-Det}

\author*[1]{\fnm{Josu} \sur{Zabala Muxika}}\email{jzabala@mondragon.edu}
\author[1]{\fnm{Javier} \sur{Solana Crespo}}\email{jsolana@mondragon.edu}

\affil*[1]{\orgdiv{Facultad de Ingeniería}, \orgname{Mondragon Unibertsitatea}, \orgaddress{\city{Arrasate-Mondragón}, \state{Gipuzkoa}, \country{España}}}

\abstract{Se estudia el modelo M2F2-Det (\emph{Multi-Modal Interpretable Forged Face Detector}, CVPR 2025 Oral), que combina CLIP ViT-L/14, DenseNet-121 y LLaVA-7B para detectar y explicar deepfakes. Se implementa el pipeline en Google Colab, obteniendo AUC\,=\,100\,\% y accuracy\,=\,99,9\,\% sobre datos GAN. Se diagnostica el fallo del modelo ante imágenes de difusión y se propone un fine-tuning con 9\,000 rostros de Stable Diffusion. Tras el reentrenamiento, el modelo detecta correctamente imágenes generadas por difusión con 100\,\% de confianza.}

\keywords{Deepfake Detection, CLIP, Vision-Language Models, Trustworthy AI, Diffusion Models, Fine-Tuning}

\maketitle

\begin{center}
\includegraphics[height=2.5cm]{663-6638495_mondragon-unibertsitatea-logo-hd-png-download.png}
\end{center}

%% ═══════════════════════════════════════════════════════════════════════
\section{Introducción}\label{sec:intro}
%% ═══════════════════════════════════════════════════════════════════════

El término \emph{deepfake} designa contenido multimedia generado o manipulado mediante aprendizaje profundo para suplantar identidades o crear contenido engañoso. Desde las redes generativas adversarias (GANs)~\cite{goodfellow2014generative}, los deepfakes han evolucionado con los modelos de difusión~\cite{rombach2022latentdiffusion}, capaces de producir imágenes fotorrealistas sin precedentes.

La detección automática de deepfakes es un pilar de la \textbf{IA Confiable}, ya que la desinformación visual puede afectar a procesos democráticos y erosionar la confianza pública. La UE, a través del AI Act, clasifica los sistemas de generación de contenido sintético como de alto riesgo~\cite{euaiact2024}.

Los detectores tradicionales basados en CNNs presentan dos limitaciones fundamentales: (1)~generalización limitada a técnicas no vistas, y (2)~falta de explicabilidad. Para abordar ambos problemas, Guo et~al.~\cite{guo2025m2f2det} proponen \textbf{M2F2-Det}, un detector multimodal que combina CLIP~\cite{radford2021clip}, DenseNet-121 y un LLM (LLaVA-7B) para ofrecer detección y explicación textual.

En este trabajo se implementa M2F2-Det en Google Colab, se analiza su rendimiento sobre deepfakes GAN, se diagnostica su fallo ante imágenes de difusión y se propone un fine-tuning con datos de Stable Diffusion que resuelve esta limitación.


%% ═══════════════════════════════════════════════════════════════════════
\section{Fundamentos Teóricos}\label{sec:background}
%% ═══════════════════════════════════════════════════════════════════════

\subsection{CLIP}\label{subsec:clip}

CLIP (\emph{Contrastive Language-Image Pre-training})~\cite{radford2021clip} es un modelo multimodal de OpenAI que aprende representaciones conjuntas de imágenes y texto mediante aprendizaje contrastivo sobre 400 millones de pares imagen-texto. Consta de un Vision Transformer (ViT)~\cite{dosovitskiy2020vit} y un Transformer textual~\cite{vaswani2017attention}, entrenados para maximizar la similitud coseno entre pares correspondientes mediante la pérdida InfoNCE.

En detección de deepfakes, CLIP aporta representaciones semánticas que capturan la relación entre la apariencia visual y descripciones como ``a real face'' o ``a manipulated face''. La variante utilizada es \textbf{ViT-L/14@336} (24 capas, embeddings de 1024 dimensiones). Sin embargo, estas representaciones pueden sesgar la predicción hacia ``real'' cuando la imagen generada es visualmente coherente, como ocurre con los modelos de difusión.

\subsection{Modelos de difusión}\label{subsec:diffusion}

Los modelos de difusión~\cite{rombach2022latentdiffusion, ho2020ddpm} generan imágenes mediante eliminación iterativa de ruido, partiendo de ruido gaussiano puro. \textbf{Stable Diffusion} opera en un espacio latente comprimido; sus variantes (SD~1.5 a $512\times512$, SD~2.1 a $768\times768$ y SDXL a $1024\times1024$) producen artefactos distintos a los de las GANs: texturas más suaves y naturales en lugar de patrones espectrales de alta frecuencia, lo que dificulta su detección por métodos entrenados exclusivamente con datos GAN.

\subsection{DenseNet-121}\label{subsec:cnn}

DenseNet-121~\cite{huang2017densenet} conecta cada capa con todas las siguientes mediante conexiones densas, favoreciendo la reutilización de características. En M2F2-Det actúa como \emph{deepfake encoder}, extrayendo características de textura y frecuencia que complementan las representaciones semánticas de CLIP.


%% ═══════════════════════════════════════════════════════════════════════
\section{Arquitectura de M2F2-Det}\label{sec:architecture}
%% ═══════════════════════════════════════════════════════════════════════

M2F2-Det~\cite{guo2025m2f2det} opera en tres etapas:

\subsection{Stage 1: Detector binario}\label{subsec:stage1}

Combina tres ramas: (1)~\textbf{CLIP-Vision}: ViT-L/14@336 congelado produce un token CLS ($\mathbf{v}_{\text{cls}} \in \mathbb{R}^{1024}$) escalado por $\alpha_{\text{vis}}$; (2)~\textbf{CLIP-Text}: embeddings textuales de prompts de detección, que generan puntuaciones de similitud escaladas por $\alpha_{\text{txt}}$; (3)~\textbf{DenseNet-121}: red entrenable que extrae características de textura, fusionadas con CLIP mediante un \textbf{Bridge Adapter}. Las tres salidas se concatenan:
\begin{equation}
    \mathbf{f} = [\alpha_{\text{vis}} \cdot \mathbf{v}_{\text{cls}} \;\|\; \alpha_{\text{txt}} \cdot \mathbf{c}_{\text{bridge}} \;\|\; \text{proj}(\text{AvgPool}(\mathbf{d}))]
\end{equation}
produciendo $\mathbf{f} \in \mathbb{R}^{2176}$ que pasa por $\text{Linear}(2176, 2)$ para clasificación binaria.

\subsection{Stage 2 y 3: Alineación y explicación}\label{subsec:stage23}

Stage~2 alinea los embeddings del detector con el espacio de LLaVA-v1.5-7B~\cite{liu2023llava} mediante un projector multimodal. Stage~3 utiliza el LLM completo para generar explicaciones textuales sobre la autenticidad de la imagen, describiendo indicios visuales como textura de piel, simetría y bordes.

En nuestros experimentos, Stage~1 clasificó una imagen manipulada como \textbf{FAKE} (78,8\,\% confianza), mientras que Stage~3 la clasificó erróneamente como real con una explicación convincente pero incorrecta. Esta discrepancia ilustra el sesgo semántico de los LLMs.


%% ═══════════════════════════════════════════════════════════════════════
\section{Metodología Experimental}\label{sec:methodology}
%% ═══════════════════════════════════════════════════════════════════════

Todo el trabajo se realizó en notebooks de Google Colab (GPUs T4/A100) conectados desde VS Code.

\subsection{Pipeline de inferencia}\label{subsec:colab}

El primer notebook (19 celdas) implementa el pipeline completo: setup del entorno con parches de compatibilidad (cuantización 4-bit, fallback de flash attention), descarga de pesos (${\sim}$16\,GB totales), y ejecución de Stage~1 (detección binaria) y Stage~3 (detección + explicación con LLaVA-7B en 4-bit).

\subsection{Entrenamiento y fine-tuning}\label{subsec:finetune}

El segundo notebook (32 celdas) implementa:

\textbf{Entrenamiento v1 (datos GAN):} Se utiliza el dataset de FFHQ (reales) + StyleGAN (falsos), equilibrado y dividido en train (80\,\%), validación (10\,\%) y test (10\,\%). Se entrena M2F2Det con DenseNet-121 (\texttt{hidden\_size=1024}), CLIP congelado, Adam (lr$\,=\,10^{-4}$), CrossEntropyLoss, batch 32, mixed precision, durante 10 épocas.

\textbf{Evaluación:} Métricas de clasificación, matriz de confusión, curva ROC, mapas Grad-CAM~\cite{selvaraju2017gradcam} sobre \texttt{denseblock4} y predicciones individuales sobre imágenes propias.

\textbf{Diagnóstico:} Al evaluar imágenes generadas por difusión (Grok/Aurora de xAI), el modelo las clasifica como \textbf{reales al 100\,\%}. El análisis de señales internas reveló que tanto CLIP (P(Real)$\,=\,$99,9\,\%) como DenseNet (P(Real)$\,=\,$97,8\,\%) fallan, confirmando que los artefactos de difusión son cualitativamente diferentes.

\textbf{Fine-tuning v2 (datos de difusión):} Se prepara el dataset \texttt{tobecwb/stable-diffusion-face-dataset}\footnote{\url{https://github.com/tobecwb/stable-diffusion-face-dataset}} con 9\,000 rostros (3\,000 de SD~1.5, 3\,000 de SD~2.1, 3\,000 de SDXL, bajo licencia CC-BY-4.0) más 4\,500 imágenes reales de FFHQ. División: 3\,600/450/450 por clase (train/val/test). Se reentrena durante 20 épocas con lr$\,=\,5\times10^{-5}$ y batch efectivo de 64.


%% ═══════════════════════════════════════════════════════════════════════
\section{Resultados}\label{sec:results}
%% ═══════════════════════════════════════════════════════════════════════

\subsection{Entrenamiento v1: Datos GAN}\label{subsec:results_v1}

La Tabla~\ref{tab:training_v1} muestra la evolución del entrenamiento v1 (StyleGAN + FFHQ).

\begin{table}[h]
\caption{Evolución del entrenamiento v1 (datos GAN).}\label{tab:training_v1}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}cccccc}
\toprule
Época & Train Loss & Train Acc & Val Loss & Val Acc & Val AUC \\
\midrule
1  & 0,2260 & 89,9\,\% & 0,0464 & 98,8\,\% & 99,93\,\% \\
3  & 0,0527 & 97,8\,\% & 0,0223 & 99,2\,\% & 99,97\,\% \\
5  & 0,0414 & 98,4\,\% & 0,0183 & 99,6\,\% & 99,98\,\% \\
10 & 0,0188 & 99,2\,\% & 0,0120 & 99,7\,\% & 99,99\,\% \\
\botrule
\end{tabular*}
\end{table}

Sobre el conjunto de test (1\,000 imágenes: 500 reales, 500 falsas GAN): Accuracy$\,=\,$99,9\,\%, AUC-ROC$\,=\,$100\,\%, F1$\,=\,$99,9\,\%, Precision$\,=\,$100\,\%, Recall$\,=\,$99,8\,\%. Solo 1 falso negativo de 500 imágenes falsas. La Figura~\ref{fig:analysis} muestra la matriz de confusión, la curva ROC y las curvas de entrenamiento.

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{notebook_exec15.png}
\caption{Resultados del entrenamiento v1 (datos GAN). Izquierda: matriz de confusión (500/500 reales correctas, 499/500 falsas correctas). Centro: curva ROC (AUC$\,=\,$1,0000). Derecha: evolución de loss y accuracy.}\label{fig:analysis}
\end{figure}

Los mapas Grad-CAM sobre \texttt{denseblock4} confirmaron que el modelo concentra su atención en la zona periocular, bordes del rostro y textura de la piel, regiones donde las GANs producen artefactos característicos.

\subsection{Predicciones individuales}\label{subsec:individual}

La Tabla~\ref{tab:individual_tests} muestra las predicciones sobre imágenes heterogéneas del modelo v1.

\begin{table}[h]
\caption{Predicciones del modelo v1 sobre imágenes de test.}\label{tab:individual_tests}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lccc}
\toprule
Imagen & Tipo real & Predicción & Confianza \\
\midrule
Imagen IA nº1 (StyleGAN) & AI-Gen & AI-Gen & 93,1\,\% \\
Imagen IA nº2 (StyleGAN) & AI-Gen & AI-Gen & 100,0\,\% \\
Foto real nº1 & Real & Real & 100,0\,\% \\
Foto real nº2 & Real & Real & 53,3\,\% \\
Imagen difusión (Grok) & AI-Gen & \textbf{Real} & \textbf{100,0\,\%} \\
\botrule
\end{tabular*}
\end{table}

El modelo detecta correctamente las imágenes GAN pero falla completamente ante imágenes de difusión, clasificándolas como reales con máxima confianza.

\subsection{Fine-tuning v2: Datos de difusión}\label{subsec:results_v2}

Tras el reentrenamiento con datos de Stable Diffusion, el modelo v2 converge rápidamente, alcanzando AUC$\,=\,$100\,\% y Accuracy$\,=\,$100\,\% desde la época 2 (Tabla~\ref{tab:training_v2}).

\begin{table}[h]
\caption{Evolución del entrenamiento v2 (datos de difusión).}\label{tab:training_v2}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}cccccc}
\toprule
Época & Train Loss & Train Acc & Val Loss & Val Acc & Val AUC \\
\midrule
1  & 0,1162 & 96,4\,\% & 0,0036 & 99,9\,\% & 100,0\,\% \\
2  & 0,0091 & 99,7\,\% & 0,0009 & 100,0\,\% & 100,0\,\% \\
5  & 0,0019 & 99,9\,\% & 0,0004 & 100,0\,\% & 100,0\,\% \\
10 & 0,0005 & 100,0\,\% & 0,0003 & 100,0\,\% & 100,0\,\% \\
20 & 0,0001 & 100,0\,\% & 0,0001 & 100,0\,\% & 100,0\,\% \\
\botrule
\end{tabular*}
\end{table}

Tras cargar los pesos del mejor checkpoint (época 2), se evaluó el modelo v2 sobre la imagen de prueba \texttt{fake\_carnet.jpg} (generada por IA). El resultado fue: \textbf{AI-GENERATED con 100,0\,\% de confianza}, resolviendo completamente el fallo de detección del modelo v1. La Figura~\ref{fig:v2pred} muestra esta predicción.

\begin{figure}[h]
\centering
\includegraphics[width=0.35\textwidth]{notebook_exec32.png}
\caption{Predicción del modelo v2 (fine-tuned con datos de difusión) sobre una imagen generada por IA. El modelo clasifica correctamente la imagen como AI-GENERATED con 100\,\% de confianza, resolviendo el fallo del modelo v1.}\label{fig:v2pred}
\end{figure}


%% ═══════════════════════════════════════════════════════════════════════
\section{Discusión}\label{sec:discussion}
%% ═══════════════════════════════════════════════════════════════════════

\subsection{Gap de dominio GAN-difusión}

Los resultados confirman que la arquitectura M2F2-Det es altamente efectiva para detectar deepfakes GAN (AUC$\,=\,$100\,\%), pero el detector aprende artefactos específicos de la técnica de generación, no características universales de artificialidad. Las GANs producen patrones espectrales característicos (artefactos de checkerboard, inconsistencias en altas frecuencias) que DenseNet-121 detecta eficazmente. Los modelos de difusión generan texturas más naturales mediante refinamiento iterativo, con artefactos cualitativamente diferentes.

El análisis de señales internas confirmó que ambas ramas fallan: CLIP porque su entrenamiento contrastivo produce representaciones semánticas (no forenses), y DenseNet porque sus filtros se especializaron en artefactos GAN. Los parámetros $\alpha$ aprendidos ($\alpha_{\text{vis}}\,=\,0{,}6244$, $\alpha_{\text{txt}}\,=\,4{,}4632$) indican que el modelo otorga mucho más peso a la rama textual, amplificando el sesgo semántico.

\subsection{Eficacia del fine-tuning y limitaciones}

El fine-tuning v2 resuelve completamente el fallo de detección de imágenes de difusión, con convergencia muy rápida (AUC$\,=\,$100\,\% en 2 épocas). Esto indica que los artefactos de difusión, aunque diferentes a los de GAN, son igualmente discriminables una vez incluidos en los datos de entrenamiento.

Sin embargo, existen limitaciones conocidas: (1)~el modelo v2 podría especializarse en Stable Diffusion y fallar ante otros generadores (DALL-E, Midjourney, Flux); (2)~existe riesgo de \emph{catastrophic forgetting} sobre datos GAN; (3)~el dataset de 4\,500 imágenes por clase es relativamente pequeño. Además, las explicaciones del LLM (Stage~3) pueden ser convincentes pero incorrectas, planteando riesgos de \emph{falsa confianza} que subrayan la necesidad de nunca confiar exclusivamente en la explicación textual.


%% ═══════════════════════════════════════════════════════════════════════
\section{Conclusiones}\label{sec:conclusion}
%% ═══════════════════════════════════════════════════════════════════════

Se ha implementado y evaluado M2F2-Det para la detección multimodal de deepfakes, desplegando el pipeline completo en Google Colab. El modelo entrenado con datos GAN alcanza un rendimiento excepcional (AUC$\,=\,$100\,\%, Accuracy$\,=\,$99,9\,\%), pero falla ante imágenes de difusión. Mediante análisis de señales internas se diagnosticó que tanto CLIP como DenseNet fallan ante estos artefactos. El fine-tuning con 9\,000 imágenes de Stable Diffusion (1.5/2.1/SDXL) resuelve completamente esta limitación, alcanzando AUC$\,=\,$100\,\% y detectando imágenes de difusión con 100\,\% de confianza. Estos resultados demuestran la eficacia de la fusión CLIP\,+\,DenseNet\,+\,Bridge Adapter y la importancia de incluir datos representativos de cada técnica de generación en el entrenamiento.

%% ═══════════════════════════════════════════════════════════════════════

\backmatter

\begin{thebibliography}{99}

\bibitem{guo2025m2f2det}
Guo, X., Zhang, Y., Liu, J., Lyu, S.:
Rethinking vision-language model in face forensics: Multi-modal interpretable forged face detector.
In: CVPR, pp.~1--12 (2025).
\url{https://arxiv.org/abs/2503.20188}

\bibitem{radford2021clip}
Radford, A., Kim, J.W., Hallacy, C., et~al.:
Learning transferable visual models from natural language supervision.
In: ICML, pp.~8748--8763. PMLR (2021)

\bibitem{liu2023llava}
Liu, H., Li, C., Wu, Q., Lee, Y.J.:
Visual instruction tuning.
In: NeurIPS, vol.~36 (2023)

\bibitem{goodfellow2014generative}
Goodfellow, I., Pouget-Abadie, J., Mirza, M., et~al.:
Generative adversarial nets.
In: NeurIPS, vol.~27 (2014)

\bibitem{rombach2022latentdiffusion}
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.:
High-resolution image synthesis with latent diffusion models.
In: CVPR, pp.~10684--10695 (2022)

\bibitem{ho2020ddpm}
Ho, J., Jain, A., Abbeel, P.:
Denoising diffusion probabilistic models.
In: NeurIPS, vol.~33, pp.~6840--6851 (2020)

\bibitem{huang2017densenet}
Huang, G., Liu, Z., Van~Der~Maaten, L., Weinberger, K.Q.:
Densely connected convolutional networks.
In: CVPR, pp.~4700--4708 (2017)

\bibitem{dosovitskiy2020vit}
Dosovitskiy, A., Beyer, L., Kolesnikov, A., et~al.:
An image is worth 16x16 words: Transformers for image recognition at scale.
In: ICLR (2021)

\bibitem{vaswani2017attention}
Vaswani, A., Shazeer, N., Parmar, N., et~al.:
Attention is all you need.
In: NeurIPS, vol.~30 (2017)

\bibitem{selvaraju2017gradcam}
Selvaraju, R.R., Cogswell, M., Das, A., et~al.:
Grad-CAM: Visual explanations from deep networks via gradient-based localization.
In: ICCV, pp.~618--626 (2017)

\bibitem{euaiact2024}
European Parliament and Council of the European Union:
Regulation (EU) 2024/1689 --- Artificial Intelligence Act.
Official Journal of the EU, L series (2024)

\end{thebibliography}

\end{document}
"""

with open("Deepfakes.tex", "w", encoding="utf-8") as f:
    f.write(content)

lines = content.count("\n")
print(f"Written {len(content)} chars, {lines} lines")
