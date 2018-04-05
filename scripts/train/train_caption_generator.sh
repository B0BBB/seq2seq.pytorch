DATASET='CocoCaptions'
DATASET_DIR=${1:-"/home/olegzendel@st.technion.ac.il/project/seq2seq.pytorch/data/COCO"}
OUTPUT_DIR=${2:-"./results"}

python main.py \
  --save captions_resnet50 \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --results_dir ${OUTPUT_DIR} \
  --model Img2Seq \
  --model_config "{'encoder': {'model': 'resnet50', 'finetune': False, 'context_transform': 256 }, \
                   'decoder': {'num_layers': 2, 'hidden_size': 256, 'dropout': 0, \
                               'tie_embedding': False,\
                               'attention': {'mode': 'bahdanau', 'normalize': True}}}" \
  --data_config "{'tokenization':'bpe', 'num_symbols':8000, 'mark_language': True, 'shared_vocab':True}" \
  --b 128 \
  --epochs 10\
  --trainer Img2SeqTrainer \
  --optimization_config "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
                          {'epoch': 4, 'optimizer': 'Adam', 'lr': 1e-4},
                          {'epoch': 8, 'optimizer': 'SGD', 'lr': 1e-4, 'momentum': 0.9}]"
