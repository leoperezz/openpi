# Fine-tuning de pi05 con LeRobot

Desde la raíz del repositorio:

```bash
bash bin/install.sh
uv run wandb login
bash bin/train.sh NONHUMAN-RESEARCH/hanoi-rewards-triple
```

La configuración está en `configs/train.yaml`. El instalador usa Python 3.11
(`.python-version`) y `uv.lock`; instala JAX/CUDA 12, OpenPI, PyAV y las demás
dependencias. Se necesita Linux y un driver NVIDIA compatible con CUDA 12.
PyAV incluye sus bibliotecas de FFmpeg; no hace falta instalar FFmpeg con sudo.
El instalador no configura credenciales: también puedes usar `WANDB_API_KEY` y,
para datasets privados, `HF_TOKEN`.

El entrenamiento usa **OpenPI/JAX**, con fine-tuning completo de pi05 en bfloat16
y pesos oficiales `gs://openpi-assets/checkpoints/pi05_base/params`. El checkpoint
`lerobot/pi05_base` del otro proyecto no se carga con este backend. Se conserva
el espacio absoluto de acciones del dataset, sin conversiones ALOHA ni deltas.
Los campos reward/success no forman parte de la pérdida: es aprendizaje por
imitación con el objetivo de flow matching de pi05, no reinforcement learning.

La configuración inicial usa batch global 8. Requiere una GPU con bastante VRAM;
el consumo depende del batch y las cámaras. No se ha validado el entrenamiento
completo con este batch en esta máquina. Selecciona una sola GPU en `configs/train.yaml`:

```yaml
device: cuda:1 # Segunda GPU; usa cuda:0 para la primera.
fsdp_devices: 1
```

También puedes elegirla por CLI con `--set device=cuda:0`. El launcher configura
`CUDA_VISIBLE_DEVICES` antes de importar JAX/PyTorch y reemplaza cualquier valor
heredado de esa variable. El número indica el ordinal CUDA del host en orden PCI
(`CUDA_DEVICE_ORDER=PCI_BUS_ID`); dentro del proceso JAX la verá como GPU 0.
`batch_size` corresponde íntegramente a esa GPU. Se entrenan todos los parámetros
del modelo. `fsdp_devices` debe permanecer en 1.

## Parámetros

```bash
bash bin/train.sh NONHUMAN-RESEARCH/hanoi-rewards-triple \
  --set batch_size=16 \
  --set num_workers=8 \
  --set data_ratio=1.5 \
  --set optimizer.learning_rate=4.0e-5 \
  --set wandb.project=ocbench \
  --set wandb.entity=mu-robotics \
  --set job_name=hanoi-pi05-run2
```

La duración se controla con `data_ratio: 1.5`: el presupuesto es
`ceil(data_ratio * frames_train / batch_size)` pasos. Solo cuenta el split train,
sin los frames reservados para validación. Por ejemplo, 1000 frames de train y
batch 8 producen 188 pasos con ratio 1.5 (1504 muestras procesadas).
El redondeo añade menos de un batch. Son épocas efectivas por cantidad de muestras:
el loader mezcla los datos en cada vuelta y descarta el batch incompleto, por lo
que no garantiza visitar cada frame exactamente una vez por época efectiva.

`data_ratio` acepta cualquier número positivo y finito, incluidas fracciones
menores que 1. Tiene prioridad sobre `num_train_steps`; para usar pasos fijos:
`--set data_ratio=null --set num_train_steps=30000`. Los YAML antiguos sin
`data_ratio` conservan su comportamiento. El total calculado aparece en los logs,
en la configuración inicial de `run.json` y en W&B. También se calcula con
`--prepare-only`. Al reanudar, el objetivo es total, no adicional: pasar de ratio
1.5 a 3.0 continúa hasta 3.0 épocas efectivas contando los pasos ya guardados.
El warmup y el decay del optimizador siguen expresados en pasos; ajusta sus
valores si eliges una duración corta.

También admite `--config otra.yaml`. `--set clave=valor` modifica una clave
existente del YAML; los valores usan sintaxis YAML. Para ejecución sin red en W&B
usa `--set wandb.mode=offline`; para desactivarlo, `--set wandb.enable=false`.
W&B registra pérdida train/val, mejor pérdida, learning rate y norma del gradiente.

Se descarga una revisión concreta del dataset y se registra su commit. El lector
soporta LeRobot v2/v3 con estados/acciones vectoriales de hasta 32 dimensiones y
cámaras de video. Usa `dataset.root` para un dataset local. Para otro robot cambia
`state_key`, `action_key` y el mapa `cameras` según `meta/info.json`.
Las claves del mapa son los tres slots de pi05; los slots ausentes se enmascaran.
En Hanoi se asignan front, side y wrist, respectivamente, a esos slots; sus nombres
internos no implican que el robot tenga dos cámaras de muñeca.

## Train / val y normalización

- Separación reproducible por **episodios completos**, 90/10 y semilla 42.
- Con los 525 episodios de Hanoi: 473 train, 52 val.
- Chunks de 50 acciones; al final del episodio se repite la última acción.
- Estadísticas de estado/acción calculadas exclusivamente sobre train,
  guardadas y reutilizadas al reanudar. Normalización por cuantiles de pi05.
- Validación cada 400 pasos y al terminar, sin aumentos de entrenamiento.
  Usa hasta 2048 frames, elegidos una vez con semilla fija de todo el split val.
  `max_eval_samples: null` evalúa todos los frames de val.
- Las muestras de ruido/tiempo se fijan entre evaluaciones para comparar pérdidas.
  El último batch se rellena y el relleno no cuenta para la métrica.
- `best` significa menor **pérdida de flow matching en validación**; no tasa de éxito
  del robot. No se ejecutan rollouts en un entorno.

Para descargar y comprobar los datos y calcular las estadísticas sin cargar pi05:

```bash
bash bin/train.sh NONHUMAN-RESEARCH/hanoi-rewards-triple --prepare-only
# Después de preparar, el directorio ya existe:
bash bin/train.sh NONHUMAN-RESEARCH/hanoi-rewards-triple --resume
```

## Checkpoints

```text
outputs/training/hanoi-rewards-triple-pi05/
  run.json                  # configuración original, revisión y episodios del split
  assets/lerobot/            # estadísticas de train
  wandb_id.txt
  last/<paso>/              # último checkpoint completo; retiene solo uno
    params/                 # pesos para inferencia
    train_state/            # parámetros + optimizador + contador de pasos
    assets/lerobot/
  best/<paso>/              # mejor checkpoint; retiene solo uno
    params/
    train_state/
    assets/lerobot/
    metrics.json            # paso y pérdida val
```

Se guarda `last` cada `save_steps` y al finalizar normalmente. `best` se actualiza
solo cuando mejora val. En una interrupción se conserva el último guardado
periódico completo; los pasos posteriores no están garantizados.

```bash
bash bin/train.sh NONHUMAN-RESEARCH/hanoi-rewards-triple --resume
# Continuar más allá del objetivo original:
bash bin/train.sh NONHUMAN-RESEARCH/hanoi-rewards-triple --resume --set data_ratio=3.0
```

Al reanudar se recuperan pesos, optimizador, paso, normalización, mejor pérdida
y el ID de W&B. El orden del dataloader vuelve a empezar: la continuación no es
idéntica bit a bit a una ejecución sin interrupciones. La revisión y el split se
mantienen; no modifiques un dataset local durante una ejecución. Cambios de
modelo, batch, dataset, evaluación u optimizador requieren un `job_name` nuevo.
Aumentar `data_ratio` o `num_train_steps` no cambia la curva de learning rate original.
No se sobrescriben ejecuciones existentes ni se suben pesos automáticamente a HF.

Los checkpoints son Orbax/OpenPI, no el formato `save_pretrained` de LeRobot.
Para inferencia hay que conservar el mapa de cámaras, las estadísticas del
checkpoint y recortar la salida a las dimensiones de acción del robot (7 en Hanoi).

## Verificación local

```bash
JAX_PLATFORMS=cpu uv run pytest -q scripts/train_pi05_test.py
```
